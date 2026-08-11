//! GPU-pixel acceptance harness for retained UI.

extern crate alloc;

use alloc::sync::Arc;
use bevy::ui_render::RenderUiSystems;
use bevy::ui_render::{UiRenderInfrastructurePlugin, UiRenderPlugin};
use bevy::{
    asset::{AssetId, RenderAssetUsages},
    camera::{ClearColorConfig, RenderTarget, Viewport},
    input_focus::InputFocus,
    log::LogPlugin,
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssetBytesPerFrame,
        render_resource::{Extent3d, PollType, TextureDimension, TextureFormat, TextureUsages},
        renderer::RenderDevice,
        ExtractSchedule, RenderApp, RenderPlugin,
    },
    text::{EditableText, TextCursorStyle, TextEdit, TextLayoutInfo},
    window::{ExitCondition, WindowPlugin},
};
use bevy_ui_render_retained::{
    RetainedUiLayerCounters, RetainedUiLayerWork, RetainedUiPaintCounters, RetainedUiRenderPlugin,
    WorkCounters,
};
use core::time::Duration;
use std::{
    fs::OpenOptions,
    sync::{Mutex, PoisonError},
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const BYTES_PER_PIXEL: usize = 4;

static GPU_TEST_MUTEX: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy)]
enum PaintSchedule {
    EveryFrame,
    UntilInitialCapture,
    UntilInitialCaptureThenDisableCamera,
}

#[derive(Clone, Copy)]
enum UiRenderer {
    Stock,
    Retained,
}

#[derive(Resource)]
struct PaintEnabled(bool);

fn paint_enabled(enabled: Res<PaintEnabled>) -> bool {
    enabled.0
}

fn with_gpu_lock(test: impl FnOnce()) {
    let _process_guard = GPU_TEST_MUTEX
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let lock_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(std::env::temp_dir().join("bevy-retained-ui-gpu-test.lock"))
        .expect("GPU test lock file must open");
    lock_file.lock().expect("GPU tests must run serially");
    test();
}

fn step_and_wait(app: &mut App) {
    app.update();
    app.world()
        .resource::<RenderDevice>()
        .wgpu_device()
        .poll(PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        })
        .expect("GPU work must complete");
}

fn capture_fresh(app: &mut App, pixels: &Arc<Mutex<Option<Vec<u8>>>>) -> Vec<u8> {
    for _ in 0..3 {
        step_and_wait(app);
    }
    pixels.lock().unwrap_or_else(PoisonError::into_inner).take();

    for _ in 0..20 {
        step_and_wait(app);
        if let Some(pixels) = pixels.lock().unwrap_or_else(PoisonError::into_inner).take() {
            return pixels;
        }
    }

    panic!("no GPU readback completed after 20 drained frames");
}

struct RenderOutput {
    pixels: Vec<u8>,
    before_mutation: Option<RetainedUiLayerWork>,
    after_mutation: Option<RetainedUiLayerWork>,
    paint_before_mutation: Option<WorkCounters>,
    paint_after_mutation: Option<WorkCounters>,
}

fn layer_work(app: &App) -> Option<RetainedUiLayerWork> {
    app.sub_app(RenderApp)
        .world()
        .get_resource::<RetainedUiLayerCounters>()
        .map(RetainedUiLayerCounters::snapshot)
}

fn paint_work(app: &App) -> Option<WorkCounters> {
    app.sub_app(RenderApp)
        .world()
        .get_resource::<RetainedUiPaintCounters>()
        .map(RetainedUiPaintCounters::snapshot)
}

fn render_scene<S>(
    renderer: UiRenderer,
    paint_schedule: PaintSchedule,
    setup: impl FnOnce(&mut World, Entity) -> S,
    mutate: impl FnOnce(&mut World, S),
) -> RenderOutput {
    let mut app = App::new();
    let mut default_plugins = DefaultPlugins
        .set(WindowPlugin {
            primary_window: None,
            exit_condition: ExitCondition::DontExit,
            ..default()
        })
        .set(RenderPlugin {
            synchronous_pipeline_compilation: true,
            ..default()
        })
        .disable::<LogPlugin>();
    if matches!(renderer, UiRenderer::Retained) {
        default_plugins = default_plugins.disable::<UiRenderPlugin>();
    }
    app.add_plugins(default_plugins);
    if matches!(renderer, UiRenderer::Retained) {
        app.add_plugins((UiRenderInfrastructurePlugin, RetainedUiRenderPlugin));
    }
    if matches!(
        paint_schedule,
        PaintSchedule::UntilInitialCapture | PaintSchedule::UntilInitialCaptureThenDisableCamera
    ) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(PaintEnabled(true));
        render_app.configure_sets(
            ExtractSchedule,
            RenderUiSystems::ExtractBackgrounds.run_if(paint_enabled),
        );
    }

    let mut image = Image::new_fill(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC
        | TextureUsages::RENDER_ATTACHMENT;
    let image = app.world_mut().resource_mut::<Assets<Image>>().add(image);

    let camera = app
        .world_mut()
        .spawn((
            Camera2d,
            Camera {
                clear_color: match paint_schedule {
                    PaintSchedule::EveryFrame => ClearColorConfig::Custom(Color::BLACK),
                    PaintSchedule::UntilInitialCapture
                    | PaintSchedule::UntilInitialCaptureThenDisableCamera => ClearColorConfig::None,
                },
                ..default()
            },
            RenderTarget::Image(image.clone().into()),
        ))
        .id();
    let scene = setup(app.world_mut(), camera);

    let pixels = Arc::new(Mutex::new(None));
    let observer_pixels = Arc::clone(&pixels);
    app.world_mut()
        .spawn(Readback::texture(image))
        .observe(move |event: On<ReadbackComplete>| {
            *observer_pixels
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Some(event.data.clone());
        });

    app.finish();
    app.cleanup();

    for _ in 0..20 {
        step_and_wait(&mut app);
    }
    capture_fresh(&mut app, &pixels);
    let before_mutation = layer_work(&app);
    let paint_before_mutation = paint_work(&app);
    if matches!(
        paint_schedule,
        PaintSchedule::UntilInitialCapture | PaintSchedule::UntilInitialCaptureThenDisableCamera
    ) {
        app.sub_app_mut(RenderApp)
            .world_mut()
            .resource_mut::<PaintEnabled>()
            .0 = false;
    }
    if matches!(
        paint_schedule,
        PaintSchedule::UntilInitialCaptureThenDisableCamera
    ) {
        app.world_mut()
            .entity_mut(camera)
            .get_mut::<Camera>()
            .unwrap()
            .is_active = false;
    }
    mutate(app.world_mut(), scene);
    let pixels = capture_fresh(&mut app, &pixels);
    let after_mutation = layer_work(&app);
    let paint_after_mutation = paint_work(&app);
    RenderOutput {
        pixels,
        before_mutation,
        after_mutation,
        paint_before_mutation,
        paint_after_mutation,
    }
}

fn assert_pixels_eq(actual: &[u8], expected: &[u8]) {
    assert_eq!(actual.len(), expected.len());
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
    {
        let pixel = index / BYTES_PER_PIXEL;
        let byte = pixel * BYTES_PER_PIXEL;
        panic!(
            "pixels first differ at ({}, {}): actual {:?}, expected {:?}",
            pixel as u32 % WIDTH,
            pixel as u32 / WIDTH,
            &actual[byte..byte + BYTES_PER_PIXEL],
            &expected[byte..byte + BYTES_PER_PIXEL]
        );
    }
}

fn spawn_full_background(world: &mut World, camera: Entity, color: Color) -> Entity {
    world
        .spawn((
            Node {
                width: percent(100),
                height: percent(100),
                ..default()
            },
            BackgroundColor(color),
            UiTargetCamera(camera),
        ))
        .id()
}

fn spawn_leaf_background(world: &mut World, camera: Entity, node: Node) -> Entity {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    world
        .spawn((
            node,
            BackgroundColor(Color::srgb_u8(220, 45, 28)),
            ChildOf(root),
        ))
        .id()
}

fn add_solid_image(world: &mut World, color: [u8; 4]) -> Handle<Image> {
    world.resource_mut::<Assets<Image>>().add(Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &color,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    ))
}

fn add_color_strip(world: &mut World) -> Handle<Image> {
    world.resource_mut::<Assets<Image>>().add(Image::new_fill(
        Extent3d {
            width: 8,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[
            220, 45, 28, 255, 20, 190, 80, 255, 30, 70, 210, 255, 190, 150, 20, 255, 80, 30, 160,
            255, 15, 175, 210, 255, 210, 60, 150, 255, 100, 120, 140, 255,
        ],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    ))
}

fn spawn_sampled_strip(world: &mut World, camera: Entity) -> Handle<Image> {
    let image = add_color_strip(world);
    world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(8),
            top: px(9),
            width: px(20),
            height: px(10),
            ..default()
        },
        ImageNode::new(image.clone())
            .with_rect(Rect::from_corners(Vec2::new(1.0, 0.0), Vec2::new(3.0, 1.0)))
            .with_mode(NodeImageMode::Stretch),
        UiTargetCamera(camera),
    ));
    image
}

fn spawn_image_leaf(
    world: &mut World,
    camera: Entity,
    image: Handle<Image>,
    tint: Color,
) -> Entity {
    world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(8),
                top: px(9),
                width: px(10),
                height: px(10),
                ..default()
            },
            ImageNode::new(image)
                .with_mode(NodeImageMode::Stretch)
                .with_color(tint),
            UiTargetCamera(camera),
        ))
        .id()
}

fn border_node() -> Node {
    Node {
        position_type: PositionType::Absolute,
        left: px(8),
        top: px(9),
        width: px(20),
        height: px(20),
        border: UiRect::all(px(4)),
        border_radius: BorderRadius::all(px(6)),
        ..default()
    }
}

fn spawn_border_leaf(
    world: &mut World,
    camera: Entity,
    colors: BorderColor,
    background: bool,
) -> Entity {
    if background {
        let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
        world.spawn((border_node(), colors, ChildOf(root))).id()
    } else {
        world
            .spawn((border_node(), colors, UiTargetCamera(camera)))
            .id()
    }
}

fn spawn_text_leaf(world: &mut World, camera: Entity, color: Color) -> Entity {
    world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(4),
                top: px(18),
                width: px(56),
                height: px(24),
                ..default()
            },
            Text::new("Retained"),
            TextColor(color),
            TextFont {
                font_size: FontSize::Px(18.0),
                ..default()
            },
            UiTargetCamera(camera),
        ))
        .id()
}

fn spawn_shadowed_text(
    world: &mut World,
    camera: Entity,
    text_color: Color,
    shadow: TextShadow,
) -> Entity {
    let text = spawn_text_leaf(world, camera, text_color);
    world.entity_mut(text).insert(shadow);
    text
}

fn spawn_decorated_text(world: &mut World, camera: Entity, underline: Color) -> Entity {
    let text = spawn_shadowed_text(
        world,
        camera,
        Color::srgb_u8(235, 210, 80),
        TextShadow {
            offset: Vec2::new(2.0, 2.0),
            color: Color::srgb_u8(35, 145, 185),
        },
    );
    world.entity_mut(text).insert((
        TextBackgroundColor(Color::srgba_u8(95, 35, 130, 180)),
        Strikethrough,
        StrikethroughColor(Color::srgb_u8(225, 55, 65)),
        Underline,
        UnderlineColor(underline),
    ));
    text
}

fn spawn_selected_editable_text(world: &mut World, camera: Entity) -> Entity {
    let mut editable = EditableText::new("Retained");
    editable.queue_edit(TextEdit::SelectAll);
    let text = world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(4),
                top: px(18),
                width: px(56),
                height: px(24),
                ..default()
            },
            editable,
            TextColor(Color::srgb_u8(235, 210, 80)),
            TextFont {
                font_size: FontSize::Px(18.0),
                ..default()
            },
            TextCursorStyle {
                color: Color::srgb_u8(240, 60, 80),
                selection_color: Color::srgb_u8(40, 180, 220),
                unfocused_selection_color: Color::srgb_u8(95, 35, 130),
                selected_text_color: Some(Color::srgb_u8(25, 30, 35)),
            },
            UiTargetCamera(camera),
        ))
        .id();
    world.insert_resource(InputFocus::from_entity(text));
    text
}

fn spawn_preedit_text(world: &mut World, camera: Entity) -> Entity {
    let mut editable = EditableText::new("Retained");
    editable.queue_edit(TextEdit::ImeSetCompose {
        value: "IME".into(),
        cursor: None,
    });
    let text = world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(4),
                top: px(18),
                width: px(56),
                height: px(24),
                ..default()
            },
            editable,
            TextColor(Color::srgb_u8(235, 210, 80)),
            TextFont {
                font_size: FontSize::Px(18.0),
                ..default()
            },
            TextCursorStyle::default(),
            UiTargetCamera(camera),
        ))
        .id();
    world.insert_resource(InputFocus::from_entity(text));
    text
}

fn spawn_spanned_text(world: &mut World, camera: Entity, span_color: Color) -> Entity {
    let root = world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(4),
                top: px(18),
                width: px(56),
                height: px(24),
                ..default()
            },
            Text::new("A"),
            TextColor(Color::srgb_u8(220, 90, 35)),
            TextFont {
                font_size: FontSize::Px(18.0),
                ..default()
            },
            UiTargetCamera(camera),
        ))
        .id();

    world
        .spawn((
            TextSpan::new("B"),
            TextColor(span_color),
            TextFont {
                font_size: FontSize::Px(18.0),
                ..default()
            },
            ChildOf(root),
        ))
        .id()
}

fn first_glyph_pixel(world: &World, text: Entity) -> (AssetId<Image>, UVec3) {
    let glyph = world
        .get::<TextLayoutInfo>(text)
        .and_then(|layout| layout.glyphs.first())
        .expect("the default font must have a rasterized glyph");
    (
        glyph.atlas_info.texture,
        UVec3::new(
            glyph.atlas_info.rect.min.x.floor().max(0.0) as u32,
            glyph.atlas_info.rect.min.y.floor().max(0.0) as u32,
            0,
        ),
    )
}

fn unused_font_atlas_pixel(world: &World, text: Entity) -> (AssetId<Image>, UVec3) {
    let layout = world
        .get::<TextLayoutInfo>(text)
        .expect("text layout must exist");
    let atlas = layout
        .glyphs
        .first()
        .map(|glyph| glyph.atlas_info.texture)
        .expect("the default font must have a rasterized glyph");
    let size = world
        .resource::<Assets<Image>>()
        .get(atlas)
        .expect("font atlas image must remain in main-world assets")
        .texture_descriptor
        .size;

    for y in (0..size.height).rev() {
        for x in (0..size.width).rev() {
            let sampled = layout.glyphs.iter().any(|glyph| {
                glyph.atlas_info.texture == atlas
                    && (x as f32) >= glyph.atlas_info.rect.min.x - 1.0
                    && (x as f32) < glyph.atlas_info.rect.max.x + 1.0
                    && (y as f32) >= glyph.atlas_info.rect.min.y - 1.0
                    && (y as f32) < glyph.atlas_info.rect.max.y + 1.0
            });
            if !sampled {
                return (atlas, UVec3::new(x, y, 0));
            }
        }
    }
    panic!("font atlas must contain a pixel outside the retained glyph samples");
}

fn erase_visible_glyph_pixel(world: &mut World, text: Entity) {
    let (atlas, rect) = world
        .get::<TextLayoutInfo>(text)
        .and_then(|layout| layout.glyphs.first())
        .map(|glyph| (glyph.atlas_info.texture, glyph.atlas_info.rect))
        .expect("the default font must have a rasterized glyph");
    let mut images = world.resource_mut::<Assets<Image>>();
    let mut image = images
        .get_mut(atlas)
        .expect("font atlas image must remain in main-world assets");
    for y in rect.min.y.floor().max(0.0) as u32..rect.max.y.ceil() as u32 {
        for x in rect.min.x.floor().max(0.0) as u32..rect.max.x.ceil() as u32 {
            let pixel = image
                .pixel_bytes_mut(UVec3::new(x, y, 0))
                .expect("font atlas pixels must be CPU-readable");
            if pixel.get(3).is_some_and(|alpha| *alpha > 0) {
                pixel[3] = 0;
                return;
            }
        }
    }
    panic!("the first glyph must contain a visible atlas pixel");
}

#[test]
fn reads_pixels_drawn_by_stock_bevy_ui() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_full_background(world, camera, Color::srgb(1.0, 0.0, 0.0)),
            |_, _| {},
        );
        assert_eq!(
            output.pixels.len(),
            WIDTH as usize * HEIGHT as usize * BYTES_PER_PIXEL
        );

        let center = ((HEIGHT / 2 * WIDTH + WIDTH / 2) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &output.pixels[center..center + BYTES_PER_PIXEL],
            &[255, 0, 0, 255]
        );
    });
}

#[test]
fn a_camera_without_ui_allocates_no_retained_surface() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |_, _| {},
            |_, _| {},
        );

        let work = output.after_mutation.unwrap();
        assert_eq!(work.surfaces_created, 0);
        assert_eq!(work.repairs, 0);
        assert_eq!(work.composites, 0);
    });
}

#[test]
fn moving_an_unpainted_node_does_no_paint_work() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                world
                    .spawn((
                        Node {
                            width: px(10),
                            height: px(10),
                            ..default()
                        },
                        UiTargetCamera(camera),
                    ))
                    .id()
            },
            |world, node| {
                world
                    .entity_mut(node)
                    .get_mut::<UiTransform>()
                    .unwrap()
                    .translation = Val2::px(20, 15);
            },
        );

        assert_eq!(output.paint_after_mutation, output.paint_before_mutation);
        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
        assert_eq!(after.surfaces_created, 0);
    });
}

#[test]
fn quiet_image_pixels_match_stock_without_another_repair() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            let image = add_solid_image(world, [210, 70, 25, 255]);
            spawn_image_leaf(world, camera, image, Color::WHITE)
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            setup,
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            setup,
            |_, _| {},
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        assert!(
            retained
                .pixels
                .chunks_exact(BYTES_PER_PIXEL)
                .any(|pixel| pixel[..3] != [0, 0, 0]),
            "the default font must produce glyph pixels"
        );
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.surfaces_created, 1);
        assert_eq!(after.repairs, before.repairs);
        assert_eq!(
            retained.paint_after_mutation,
            retained.paint_before_mutation
        );
    });
}

#[test]
fn quiet_equal_color_border_matches_stock_grouping() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            spawn_border_leaf(
                world,
                camera,
                BorderColor::all(Color::srgba_u8(220, 45, 28, 190)),
                false,
            )
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            setup,
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            setup,
            |_, _| {},
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
        assert_eq!(
            retained.paint_after_mutation,
            retained.paint_before_mutation
        );
    });
}

#[test]
fn quiet_text_pixels_match_stock_without_another_repair() {
    with_gpu_lock(|| {
        let setup =
            |world: &mut World, camera| spawn_text_leaf(world, camera, Color::srgb_u8(220, 90, 35));
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            setup,
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            setup,
            |_, _| {},
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
        assert_eq!(
            retained.paint_after_mutation,
            retained.paint_before_mutation
        );
    });
}

#[test]
fn quiet_text_shadow_matches_stock_without_main_glyph_paint() {
    with_gpu_lock(|| {
        let shadow = TextShadow {
            offset: Vec2::new(3.0, 2.0),
            color: Color::srgb_u8(40, 180, 220),
        };
        let setup = move |world: &mut World, camera| {
            spawn_shadowed_text(world, camera, Color::NONE, shadow)
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            setup,
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            setup,
            |_, _| {},
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        assert!(
            retained
                .pixels
                .chunks_exact(BYTES_PER_PIXEL)
                .any(|pixel| pixel[..3] != [0, 0, 0]),
            "the text shadow must produce pixels without main glyph paint"
        );
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
    });
}

#[test]
fn changed_text_shadow_offset_repairs_old_and_new_glyph_coverage() {
    with_gpu_lock(|| {
        let initial = TextShadow {
            offset: Vec2::new(2.0, 1.0),
            color: Color::srgb_u8(40, 180, 220),
        };
        let final_shadow = TextShadow {
            offset: Vec2::new(-3.0, 2.0),
            ..initial
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            move |world, camera| {
                spawn_shadowed_text(world, camera, Color::srgb_u8(220, 90, 35), final_shadow)
            },
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            move |world, camera| {
                spawn_shadowed_text(world, camera, Color::srgb_u8(220, 90, 35), initial)
            },
            move |world, text| {
                world
                    .entity_mut(text)
                    .get_mut::<TextShadow>()
                    .unwrap()
                    .offset = final_shadow.offset;
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert!(after.repair_pixels - before.repair_pixels < 56 * 24);
    });
}

#[test]
fn quiet_text_decorations_and_their_shadows_match_stock() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            spawn_decorated_text(world, camera, Color::srgb_u8(65, 225, 105))
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            setup,
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            setup,
            |_, _| {},
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
    });
}

#[test]
fn changed_underline_color_repairs_its_text_root() {
    with_gpu_lock(|| {
        let final_color = Color::srgb_u8(55, 125, 235);
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            move |world, camera| spawn_decorated_text(world, camera, final_color),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_decorated_text(world, camera, Color::srgb_u8(65, 225, 105)),
            move |world, text| {
                world
                    .entity_mut(text)
                    .get_mut::<UnderlineColor>()
                    .unwrap()
                    .0 = final_color;
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert!(after.repair_pixels - before.repair_pixels < 56 * 24);
    });
}

#[test]
fn removing_text_background_repairs_its_vacated_run_bounds() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            spawn_decorated_text(world, camera, Color::srgb_u8(65, 225, 105))
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            setup,
            |world, text| {
                world.entity_mut(text).remove::<TextBackgroundColor>();
            },
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            setup,
            |world, text| {
                world.entity_mut(text).remove::<TextBackgroundColor>();
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
    });
}

#[test]
fn selected_editable_text_and_cursor_match_stock() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            spawn_selected_editable_text,
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            spawn_selected_editable_text,
            |_, _| {},
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        assert!(
            retained
                .pixels
                .chunks_exact(BYTES_PER_PIXEL)
                .any(|pixel| pixel[..3] != [0, 0, 0]),
            "editable glyphs and selection must produce pixels"
        );
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
    });
}

#[test]
fn changing_editable_text_focus_repairs_selection_color() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            spawn_selected_editable_text,
            |world, _| world.resource_mut::<InputFocus>().clear(),
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            spawn_selected_editable_text,
            |world, _| world.resource_mut::<InputFocus>().clear(),
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert!(after.repair_pixels - before.repair_pixels < u64::from(WIDTH * HEIGHT));
    });
}

#[test]
fn editable_preedit_underline_matches_stock() {
    with_gpu_lock(|| {
        let assert_preedit = |world: &mut World, text| {
            assert!(
                !world
                    .get::<TextLayoutInfo>(text)
                    .unwrap()
                    .preedit_underline_rects
                    .is_empty(),
                "the test must exercise preedit underline geometry"
            );
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            spawn_preedit_text,
            assert_preedit,
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            spawn_preedit_text,
            assert_preedit,
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
    });
}

#[test]
fn changed_text_color_repairs_only_glyph_coverage() {
    with_gpu_lock(|| {
        let final_color = Color::srgb_u8(40, 180, 220);
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_text_leaf(world, camera, final_color),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_text_leaf(world, camera, Color::srgb_u8(220, 90, 35)),
            move |world, text| {
                world.entity_mut(text).get_mut::<TextColor>().unwrap().0 = final_color;
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(
            after.repairs,
            before.repairs + 1,
            "layer {before:?} -> {after:?}; paint {:?} -> {:?}",
            retained.paint_before_mutation,
            retained.paint_after_mutation
        );
        assert!(after.repair_pixels - before.repair_pixels < 56 * 24);
    });
}

#[test]
fn changed_text_content_erases_vacated_glyphs() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_text_leaf(world, camera, Color::srgb_u8(220, 90, 35)),
            |world, text| {
                world.entity_mut(text).get_mut::<Text>().unwrap().0 = "UI".to_string();
            },
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_text_leaf(world, camera, Color::srgb_u8(220, 90, 35)),
            |world, text| {
                world.entity_mut(text).get_mut::<Text>().unwrap().0 = "UI".to_string();
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert!(after.repair_pixels - before.repair_pixels < 56 * 24);
    });
}

#[test]
fn changed_span_color_nominates_its_text_root() {
    with_gpu_lock(|| {
        let final_color = Color::srgb_u8(40, 180, 220);
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_spanned_text(world, camera, final_color),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_spanned_text(world, camera, Color::srgb_u8(180, 40, 210)),
            move |world, span| {
                world.entity_mut(span).get_mut::<TextColor>().unwrap().0 = final_color;
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert!(after.repair_pixels - before.repair_pixels < 56 * 24);
        let paint_before = retained.paint_before_mutation.unwrap();
        let paint_after = retained.paint_after_mutation.unwrap();
        assert_eq!(
            paint_after.records_changed,
            paint_before.records_changed + 1,
            "a span-only color change must not rewrite sibling glyph records"
        );
    });
}

#[test]
fn pending_font_atlas_upload_keeps_old_text_and_damage_owed() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_text_leaf(world, camera, Color::srgb_u8(220, 90, 35)),
            |world, text| {
                let (atlas, pixel) = first_glyph_pixel(world, text);
                world.insert_resource(RenderAssetBytesPerFrame::new(0));
                let mut images = world.resource_mut::<Assets<Image>>();
                let mut image = images
                    .get_mut(atlas)
                    .expect("font atlas image must remain in main-world assets");
                let first = image
                    .pixel_bytes_mut(pixel)
                    .ok()
                    .and_then(|pixel| pixel.first_mut())
                    .expect("font atlas must have CPU image data");
                *first = first.wrapping_add(1);
            },
        );

        assert!(
            output
                .pixels
                .chunks_exact(BYTES_PER_PIXEL)
                .any(|pixel| pixel[..3] != [0, 0, 0]),
            "the previous text pixels must remain visible"
        );
        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
        let paint_before = output.paint_before_mutation.unwrap();
        let paint_after = output.paint_after_mutation.unwrap();
        assert_eq!(
            paint_after.records_changed,
            paint_before.records_changed + 1
        );
    });
}

#[test]
fn unsampled_font_atlas_change_does_not_repaint_text() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_text_leaf(world, camera, Color::srgb_u8(220, 90, 35)),
            |world, text| {
                let (atlas, pixel) = unused_font_atlas_pixel(world, text);
                let mut images = world.resource_mut::<Assets<Image>>();
                let mut image = images
                    .get_mut(atlas)
                    .expect("font atlas image must remain in main-world assets");
                let first = image
                    .pixel_bytes_mut(pixel)
                    .ok()
                    .and_then(|pixel| pixel.first_mut())
                    .expect("font atlas must have CPU image data");
                *first = first.wrapping_add(1);
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
        assert_eq!(output.paint_after_mutation, output.paint_before_mutation);
    });
}

#[test]
fn changed_sampled_font_atlas_pixel_repairs_text() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_text_leaf(world, camera, Color::srgb_u8(220, 90, 35)),
            erase_visible_glyph_pixel,
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_text_leaf(world, camera, Color::srgb_u8(220, 90, 35)),
            erase_visible_glyph_pixel,
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        let paint_before = retained.paint_before_mutation.unwrap();
        let paint_after = retained.paint_after_mutation.unwrap();
        assert_eq!(
            paint_after.records_changed,
            paint_before.records_changed + 1
        );
    });
}

#[test]
fn one_border_edge_change_repairs_only_its_antialiased_corner_reach() {
    with_gpu_lock(|| {
        let initial = BorderColor::all(Color::srgb_u8(220, 45, 28));
        let final_left = Color::srgb_u8(180, 40, 210);
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_border_leaf(
                    world,
                    camera,
                    BorderColor {
                        left: final_left,
                        ..initial
                    },
                    true,
                )
            },
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_border_leaf(world, camera, initial, true),
            move |world, border| {
                world
                    .entity_mut(border)
                    .get_mut::<BorderColor>()
                    .unwrap()
                    .left = final_left;
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 140);
        assert_eq!(after.items_replayed, before.items_replayed + 3);
    });
}

#[test]
fn retained_outline_matches_stock_pixels() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            world
                .spawn((
                    border_node(),
                    Outline::new(px(3), px(2), Color::srgba_u8(220, 45, 28, 190)),
                    UiTargetCamera(camera),
                ))
                .id()
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            setup,
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            setup,
            |_, _| {},
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
    });
}

#[test]
fn removing_border_color_repairs_all_vacated_edge_regions() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| {
                let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
                world.spawn((border_node(), ChildOf(root))).id()
            },
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_border_leaf(
                    world,
                    camera,
                    BorderColor::all(Color::srgba_u8(220, 45, 28, 190)),
                    true,
                )
            },
            |world, border| {
                world.entity_mut(border).remove::<BorderColor>();
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 364);
    });
}

#[test]
fn image_damage_replays_the_background_beneath_it() {
    with_gpu_lock(|| {
        let final_tint = Color::srgba_u8(35, 190, 90, 170);
        let setup = move |world: &mut World, camera, tint| {
            let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
            let image = add_solid_image(world, [220, 85, 30, 210]);
            world
                .spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(8),
                        top: px(9),
                        width: px(10),
                        height: px(10),
                        ..default()
                    },
                    ImageNode::new(image)
                        .with_mode(NodeImageMode::Stretch)
                        .with_color(tint),
                    ChildOf(root),
                ))
                .id()
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, final_tint),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, Color::WHITE),
            move |world, image| {
                world
                    .entity_mut(image)
                    .get_mut::<ImageNode>()
                    .unwrap()
                    .color = final_tint;
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 100);
        assert_eq!(after.items_replayed, before.items_replayed + 2);
    });
}

#[test]
fn modified_image_asset_repairs_only_its_readers() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                let image = add_solid_image(world, [180, 25, 70, 255]);
                spawn_image_leaf(world, camera, image.clone(), Color::WHITE);
                image
            },
            |world, image| {
                world
                    .resource_mut::<Assets<Image>>()
                    .get_mut(&image)
                    .unwrap()
                    .data
                    .as_mut()
                    .unwrap()
                    .copy_from_slice(&[20, 80, 210, 255]);
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 100);
        let center = ((14 * WIDTH + 13) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &output.pixels[center..center + BYTES_PER_PIXEL],
            &[20, 80, 210, 255]
        );
    });
}

#[test]
fn unsampled_image_pixel_change_does_no_retained_work() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            spawn_sampled_strip,
            |world, image| {
                world
                    .resource_mut::<Assets<Image>>()
                    .get_mut(&image)
                    .unwrap()
                    .pixel_bytes_mut(UVec3::new(7, 0, 0))
                    .unwrap()[0] ^= 0xff;
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
        assert_eq!(after.repair_pixels, before.repair_pixels);
        assert_eq!(after.items_replayed, before.items_replayed);
        assert_eq!(output.paint_after_mutation, output.paint_before_mutation);
    });
}

#[test]
fn sampled_image_pixel_change_repairs_its_node() {
    with_gpu_lock(|| {
        let mutate = |world: &mut World, image: Handle<Image>| {
            world
                .resource_mut::<Assets<Image>>()
                .get_mut(&image)
                .unwrap()
                .pixel_bytes_mut(UVec3::new(2, 0, 0))
                .unwrap()
                .copy_from_slice(&[245, 210, 35, 255]);
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            spawn_sampled_strip,
            mutate,
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            spawn_sampled_strip,
            mutate,
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 200);
        let paint_before = retained.paint_before_mutation.unwrap();
        let paint_after = retained.paint_after_mutation.unwrap();
        assert_eq!(
            paint_after.records_changed,
            paint_before.records_changed + 1
        );
    });
}

#[test]
fn equal_image_asset_write_does_not_repaint() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                let image = add_solid_image(world, [180, 25, 70, 255]);
                spawn_image_leaf(world, camera, image.clone(), Color::WHITE);
                image
            },
            |world, image| {
                let mut images = world.resource_mut::<Assets<Image>>();
                let data = images.get(&image).unwrap().data.clone();
                images.get_mut(&image).unwrap().data = data;
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
        assert_eq!(after.repair_pixels, before.repair_pixels);
        assert_eq!(after.items_replayed, before.items_replayed);
        assert_eq!(output.paint_after_mutation, output.paint_before_mutation);
    });
}

#[test]
fn transparent_image_does_not_subscribe_to_asset_changes() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                let image = add_solid_image(world, [180, 25, 70, 255]);
                spawn_image_leaf(world, camera, image.clone(), Color::NONE);
                image
            },
            |world, image| {
                world
                    .resource_mut::<Assets<Image>>()
                    .get_mut(&image)
                    .unwrap()
                    .data
                    .as_mut()
                    .unwrap()
                    .copy_from_slice(&[20, 80, 210, 255]);
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
        assert_eq!(after.repair_pixels, before.repair_pixels);
        assert_eq!(after.items_replayed, before.items_replayed);
        assert_eq!(output.paint_after_mutation, output.paint_before_mutation);
    });
}

#[test]
fn pending_image_upload_keeps_old_pixels_and_damage_owed() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                let image = add_solid_image(world, [180, 25, 70, 255]);
                spawn_image_leaf(world, camera, image.clone(), Color::WHITE);
                image
            },
            |world, image| {
                world.insert_resource(RenderAssetBytesPerFrame::new(0));
                world
                    .resource_mut::<Assets<Image>>()
                    .get_mut(&image)
                    .unwrap()
                    .data
                    .as_mut()
                    .unwrap()
                    .copy_from_slice(&[20, 80, 210, 255]);
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
        let center = ((14 * WIDTH + 13) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &output.pixels[center..center + BYTES_PER_PIXEL],
            &[180, 25, 70, 255]
        );
        let paint_before = output.paint_before_mutation.unwrap();
        let paint_after = output.paint_after_mutation.unwrap();
        assert_eq!(
            paint_after.records_changed,
            paint_before.records_changed + 1
        );
    });
}

#[test]
fn removing_main_world_image_keeps_its_live_render_asset() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                let image = add_solid_image(world, [180, 25, 70, 255]);
                spawn_image_leaf(world, camera, image.clone(), Color::WHITE);
                image
            },
            |world, image| {
                world.resource_mut::<Assets<Image>>().remove(image.id());
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
        let center = ((14 * WIDTH + 13) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &output.pixels[center..center + BYTES_PER_PIXEL],
            &[180, 25, 70, 255]
        );
    });
}

#[test]
fn switching_to_an_unavailable_image_erases_vacated_pixels() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                let image = add_solid_image(world, [180, 25, 70, 255]);
                spawn_image_leaf(world, camera, image, Color::WHITE)
            },
            |world, entity| {
                world
                    .entity_mut(entity)
                    .get_mut::<ImageNode>()
                    .unwrap()
                    .image = Handle::from(bevy::asset::uuid::uuid!(
                    "e34373ea-19cf-4c25-b770-3c815e200379"
                ));
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 100);
        let center = ((14 * WIDTH + 13) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &output.pixels[center..center + BYTES_PER_PIXEL],
            &[0, 0, 0, 255]
        );
    });
}

fn spawn_atlas_image(world: &mut World, camera: Entity) -> (Entity, Handle<TextureAtlasLayout>) {
    let image = world.resource_mut::<Assets<Image>>().add(Image::new_fill(
        Extent3d {
            width: 2,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[220, 45, 28, 255, 20, 190, 80, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    ));
    let layout = world
        .resource_mut::<Assets<TextureAtlasLayout>>()
        .add(TextureAtlasLayout::from_grid(UVec2::ONE, 2, 1, None, None));
    let entity = world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(8),
                top: px(9),
                width: px(10),
                height: px(10),
                ..default()
            },
            ImageNode::from_atlas_image(
                image,
                TextureAtlas {
                    layout: layout.clone(),
                    index: 0,
                },
            )
            .with_mode(NodeImageMode::Stretch),
            UiTargetCamera(camera),
        ))
        .id();
    (entity, layout)
}

#[test]
fn changed_atlas_rect_repairs_its_image() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            spawn_atlas_image,
            |world, (_, layout)| {
                world
                    .resource_mut::<Assets<TextureAtlasLayout>>()
                    .get_mut(&layout)
                    .unwrap()
                    .textures[0] = URect::from_corners(UVec2::X, UVec2::new(2, 1));
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 100);
        let center = ((14 * WIDTH + 13) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &output.pixels[center..center + BYTES_PER_PIXEL],
            &[20, 190, 80, 255]
        );
    });
}

#[test]
fn irrelevant_atlas_edit_is_compared_without_repaint() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            spawn_atlas_image,
            |world, (_, layout)| {
                world
                    .resource_mut::<Assets<TextureAtlasLayout>>()
                    .get_mut(&layout)
                    .unwrap()
                    .textures[1] = URect::from_corners(UVec2::ZERO, UVec2::ONE);
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
        let paint_before = output.paint_before_mutation.unwrap();
        let paint_after = output.paint_after_mutation.unwrap();
        assert_eq!(paint_after.candidates, paint_before.candidates + 1);
        assert_eq!(
            paint_after.records_compared,
            paint_before.records_compared + 1
        );
        assert_eq!(paint_after.records_changed, paint_before.records_changed);
    });
}

#[derive(Clone, Copy)]
struct MutableScene {
    glass: Entity,
    transient: Option<Entity>,
}

fn spawn_differential_scene(world: &mut World, camera: Entity, final_state: bool) -> MutableScene {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    let glass = world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(if final_state { 31 } else { 5 }),
                top: px(if final_state { 9 } else { 29 }),
                width: px(25),
                height: px(31),
                ..default()
            },
            BackgroundColor(if final_state {
                Color::srgba_u8(220, 45, 28, 133)
            } else {
                Color::srgba_u8(20, 190, 80, 181)
            }),
            ChildOf(root),
        ))
        .id();
    let transient = (!final_state).then(|| {
        world
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: px(20),
                    top: px(14),
                    width: px(30),
                    height: px(11),
                    ..default()
                },
                BackgroundColor(Color::srgba_u8(245, 210, 30, 109)),
                ChildOf(root),
            ))
            .id()
    });
    MutableScene { glass, transient }
}

#[test]
fn full_repaint_reference_is_independent_of_mutation_history() {
    with_gpu_lock(|| {
        let direct = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_differential_scene(world, camera, true),
            |_, _| {},
        );
        let mutated = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_differential_scene(world, camera, false),
            |world, scene| {
                {
                    let mut glass = world.entity_mut(scene.glass);
                    let mut node = glass.get_mut::<Node>().unwrap();
                    node.left = px(31);
                    node.top = px(9);
                }
                world
                    .entity_mut(scene.glass)
                    .insert(BackgroundColor(Color::srgba_u8(220, 45, 28, 133)));
                world.despawn(scene.transient.unwrap());
            },
        );

        assert_pixels_eq(&mutated.pixels, &direct.pixels);
    });
}

#[test]
fn quiet_backgrounds_retain_pixels_without_another_repair() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_full_background(world, camera, Color::srgb(1.0, 0.0, 0.0)),
            |_, _| {},
        );

        let center = ((HEIGHT / 2 * WIDTH + WIDTH / 2) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &output.pixels[center..center + BYTES_PER_PIXEL],
            &[255, 0, 0, 255]
        );
        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
        assert_eq!(after.repair_pixels, before.repair_pixels);
        assert!(after.composites > before.composites);
        assert_eq!(
            output.paint_after_mutation, output.paint_before_mutation,
            "quiet Changed<T> scans must not nominate paint records"
        );
    });
}

#[test]
fn quiet_composition_touches_only_possible_content_pixels() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                world
                    .spawn((
                        Node {
                            width: px(10),
                            height: px(10),
                            ..default()
                        },
                        BackgroundColor(Color::srgb_u8(220, 45, 28)),
                        UiTargetCamera(camera),
                    ))
                    .id()
            },
            |_, _| {},
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        let composites = after.composites - before.composites;
        assert!(composites > 0);
        assert_eq!(
            after.composite_pixels - before.composite_pixels,
            composites * 100
        );
    });
}

fn spawn_disjoint_backgrounds(world: &mut World, camera: Entity) {
    let root = world
        .spawn((
            Node {
                width: percent(100),
                height: percent(100),
                ..default()
            },
            UiTargetCamera(camera),
        ))
        .id();
    for (left, color) in [
        (px(8), Color::srgba_u8(220, 45, 28, 160)),
        (px(40), Color::srgba_u8(20, 190, 80, 180)),
    ] {
        world.spawn((
            Node {
                position_type: PositionType::Absolute,
                left,
                top: px(27),
                width: px(10),
                height: px(10),
                ..default()
            },
            BackgroundColor(color),
            ChildOf(root),
        ));
    }
}

#[test]
fn disjoint_composite_regions_preserve_pixels_and_the_gap_between_them() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            spawn_disjoint_backgrounds,
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            spawn_disjoint_backgrounds,
            |_, _| {},
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let pixel = |x: u32, y: u32| {
            let start = ((y * WIDTH + x) as usize) * BYTES_PER_PIXEL;
            &stock.pixels[start..start + BYTES_PER_PIXEL]
        };
        assert_ne!(pixel(13, 32), [0, 0, 0, 255]);
        assert_eq!(pixel(32, 32), [0, 0, 0, 255]);
        assert_ne!(pixel(45, 32), [0, 0, 0, 255]);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        let composites = after.composites - before.composites;
        assert!(composites > 0);
        assert_eq!(
            after.composite_draws - before.composite_draws,
            composites * 2,
            "{before:?} -> {after:?}"
        );
        assert_eq!(
            after.composite_pixels - before.composite_pixels,
            composites * 200
        );
    });
}

#[test]
fn equal_background_write_is_nominated_but_does_not_repair() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_full_background(world, camera, Color::srgb_u8(180, 25, 70)),
            |world, background| {
                world
                    .entity_mut(background)
                    .insert(BackgroundColor(Color::srgb_u8(180, 25, 70)));
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
        assert_eq!(after.repair_pixels, before.repair_pixels);
        let paint_before = output.paint_before_mutation.unwrap();
        let paint_after = output.paint_after_mutation.unwrap();
        assert_eq!(paint_after.candidates, paint_before.candidates + 1);
        assert_eq!(
            paint_after.records_compared,
            paint_before.records_compared + 1
        );
        assert_eq!(paint_after.records_changed, paint_before.records_changed);
    });
}

#[test]
fn changed_background_encodes_exactly_one_full_repair() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_full_background(world, camera, Color::srgb_u8(180, 25, 70)),
            |world, background| {
                world
                    .entity_mut(background)
                    .insert(BackgroundColor(Color::srgb_u8(20, 80, 210)));
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(
            after.repair_pixels,
            before.repair_pixels + u64::from(WIDTH) * u64::from(HEIGHT)
        );
        let paint_before = output.paint_before_mutation.unwrap();
        let paint_after = output.paint_after_mutation.unwrap();
        assert_eq!(paint_after.candidates, paint_before.candidates + 1);
        assert_eq!(
            paint_after.records_changed,
            paint_before.records_changed + 1
        );
        let center = ((HEIGHT / 2 * WIDTH + WIDTH / 2) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &output.pixels[center..center + BYTES_PER_PIXEL],
            &[20, 80, 210, 255]
        );
    });
}

#[test]
fn moving_one_leaf_repairs_only_old_and_new_pixels_and_intersecting_items() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_leaf_background(
                    world,
                    camera,
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(5),
                        top: px(8),
                        width: px(10),
                        height: px(10),
                        ..default()
                    },
                )
            },
            |world, leaf| {
                world.entity_mut(leaf).get_mut::<Node>().unwrap().left = px(30);
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 200);
        assert_eq!(after.items_replayed, before.items_replayed + 3);

        let old_center = ((13 * WIDTH + 10) as usize) * BYTES_PER_PIXEL;
        let new_center = ((13 * WIDTH + 35) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &output.pixels[old_center..old_center + BYTES_PER_PIXEL],
            &[18, 32, 76, 255]
        );
        assert_eq!(
            &output.pixels[new_center..new_center + BYTES_PER_PIXEL],
            &[220, 45, 28, 255]
        );
    });
}

#[test]
fn ui_transform_motion_nominates_only_the_moved_leaf() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_leaf_background(
                    world,
                    camera,
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(5),
                        top: px(8),
                        width: px(10),
                        height: px(10),
                        ..default()
                    },
                )
            },
            |world, leaf| {
                world
                    .entity_mut(leaf)
                    .get_mut::<UiTransform>()
                    .unwrap()
                    .translation = Val2::px(25, 0);
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 200);
        assert_eq!(after.items_replayed, before.items_replayed + 3);
        let paint_before = output.paint_before_mutation.unwrap();
        let paint_after = output.paint_after_mutation.unwrap();
        assert_eq!(paint_after.candidates, paint_before.candidates + 1);
        assert_eq!(
            paint_after.records_changed,
            paint_before.records_changed + 1
        );
    });
}

#[test]
fn inherited_visibility_removes_only_the_hidden_leaf_pixels() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_leaf_background(
                    world,
                    camera,
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(5),
                        top: px(8),
                        width: px(10),
                        height: px(10),
                        ..default()
                    },
                )
            },
            |world, leaf| {
                world.entity_mut(leaf).insert(Visibility::Hidden);
            },
        );

        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 100);
        assert_eq!(after.items_replayed, before.items_replayed + 1);
        let hidden_center = ((13 * WIDTH + 10) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &output.pixels[hidden_center..hidden_center + BYTES_PER_PIXEL],
            &[18, 32, 76, 255]
        );
    });
}

fn spawn_clip_scene(world: &mut World, camera: Entity, clipped: bool) -> Entity {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    let clip_parent = world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(8),
                top: px(8),
                width: px(10),
                height: px(10),
                overflow: if clipped {
                    Overflow::clip()
                } else {
                    Overflow::visible()
                },
                ..default()
            },
            ChildOf(root),
        ))
        .id();
    world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(0),
            top: px(0),
            width: px(20),
            height: px(10),
            ..default()
        },
        BackgroundColor(Color::srgb_u8(220, 45, 28)),
        ChildOf(clip_parent),
    ));
    clip_parent
}

#[test]
fn removing_a_calculated_clip_rebuilds_the_newly_exposed_pixels() {
    with_gpu_lock(|| {
        let direct = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_clip_scene(world, camera, false),
            |_, _| {},
        );
        let unclipped = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_clip_scene(world, camera, true),
            |world, clip_parent| {
                world
                    .entity_mut(clip_parent)
                    .get_mut::<Node>()
                    .unwrap()
                    .overflow = Overflow::visible();
            },
        );

        assert_pixels_eq(&unclipped.pixels, &direct.pixels);
    });
}

#[test]
fn inserting_a_calculated_clip_repairs_pixels_it_hides() {
    with_gpu_lock(|| {
        let direct = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_clip_scene(world, camera, true),
            |_, _| {},
        );
        let clipped = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_clip_scene(world, camera, false),
            |world, clip_parent| {
                world
                    .entity_mut(clip_parent)
                    .get_mut::<Node>()
                    .unwrap()
                    .overflow = Overflow::clip();
            },
        );

        assert_pixels_eq(&clipped.pixels, &direct.pixels);
    });
}

fn spawn_border_scene(world: &mut World, camera: Entity, border: Val) -> Entity {
    spawn_leaf_background(
        world,
        camera,
        Node {
            position_type: PositionType::Absolute,
            left: px(8),
            top: px(8),
            width: px(20),
            height: px(20),
            border: UiRect::all(border),
            ..default()
        },
    )
}

#[test]
fn source_node_changes_nominate_bypassed_computed_paint_geometry() {
    with_gpu_lock(|| {
        let direct = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_border_scene(world, camera, px(5)),
            |_, _| {},
        );
        let mutated = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_border_scene(world, camera, px(0)),
            |world, leaf| {
                world.entity_mut(leaf).get_mut::<Node>().unwrap().border = UiRect::all(px(5));
            },
        );

        assert_pixels_eq(&mutated.pixels, &direct.pixels);
        let before = mutated.paint_before_mutation.unwrap();
        let after = mutated.paint_after_mutation.unwrap();
        assert_eq!(after.candidates, before.candidates + 1);
        assert_eq!(after.records_changed, before.records_changed + 1);
    });
}

fn spawn_outer_color_scene(world: &mut World, camera: Entity, outer: bool) -> Entity {
    let leaf = spawn_leaf_background(
        world,
        camera,
        Node {
            position_type: PositionType::Absolute,
            left: px(8),
            top: px(8),
            width: px(20),
            height: px(20),
            border_radius: BorderRadius::all(px(8)),
            ..default()
        },
    );
    if outer {
        world
            .entity_mut(leaf)
            .insert(OuterColor(Color::srgb_u8(20, 190, 80)));
    }
    leaf
}

#[test]
fn removing_outer_color_repairs_its_vacated_pixels() {
    with_gpu_lock(|| {
        let direct = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_outer_color_scene(world, camera, false),
            |_, _| {},
        );
        let removed = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_outer_color_scene(world, camera, true),
            |world, leaf| {
                world.entity_mut(leaf).remove::<OuterColor>();
            },
        );

        assert_pixels_eq(&removed.pixels, &direct.pixels);
        let before = removed.paint_before_mutation.unwrap();
        let after = removed.paint_after_mutation.unwrap();
        assert_eq!(after.records_removed, before.records_removed + 1);
    });
}

fn spawn_stack_scene(world: &mut World, camera: Entity, red_on_top: bool) -> Entity {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    let node = || Node {
        position_type: PositionType::Absolute,
        left: px(10),
        top: px(10),
        width: px(20),
        height: px(20),
        ..default()
    };
    let red = world
        .spawn((
            node(),
            ZIndex(i32::from(red_on_top)),
            BackgroundColor(Color::srgba_u8(220, 45, 28, 160)),
            ChildOf(root),
        ))
        .id();
    world.spawn((
        node(),
        BackgroundColor(Color::srgba_u8(20, 190, 80, 180)),
        ChildOf(root),
    ));
    red
}

#[test]
fn computed_stack_changes_repair_translucent_paint_order() {
    with_gpu_lock(|| {
        let direct = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_stack_scene(world, camera, true),
            |_, _| {},
        );
        let reordered = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_stack_scene(world, camera, false),
            |world, red| {
                world.entity_mut(red).insert(ZIndex(1));
            },
        );

        assert_pixels_eq(&reordered.pixels, &direct.pixels);
    });
}

#[test]
fn losing_a_renderable_target_removes_pixels_from_the_previous_camera() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_full_background(world, camera, Color::srgb_u8(220, 45, 28)),
            |world, background| {
                let non_camera = world.spawn_empty().id();
                world
                    .entity_mut(background)
                    .insert(UiTargetCamera(non_camera));
            },
        );

        let center = ((HEIGHT / 2 * WIDTH + WIDTH / 2) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &output.pixels[center..center + BYTES_PER_PIXEL],
            &[0, 0, 0, 255]
        );
        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(
            after.composites, before.composites,
            "an empty retained layer must stop compositing"
        );
    });
}

#[test]
fn removing_node_ends_ui_participation_even_if_computed_components_remain() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_full_background(world, camera, Color::srgb_u8(220, 45, 28)),
            |world, background| {
                world.entity_mut(background).remove::<Node>();
            },
        );

        let center = ((HEIGHT / 2 * WIDTH + WIDTH / 2) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &output.pixels[center..center + BYTES_PER_PIXEL],
            &[0, 0, 0, 255]
        );
        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.composites, before.composites);
    });
}

fn spawn_viewport_background(
    world: &mut World,
    camera: Entity,
    position: UVec2,
    size: UVec2,
) -> Entity {
    world
        .entity_mut(camera)
        .get_mut::<Camera>()
        .unwrap()
        .viewport = Some(Viewport {
        physical_position: position,
        physical_size: size,
        depth: 0.0..1.0,
    });
    spawn_full_background(world, camera, Color::srgb_u8(220, 45, 28))
}

#[test]
fn retained_layer_matches_a_nonzero_camera_viewport() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_viewport_background(world, camera, UVec2::new(16, 12), UVec2::new(32, 40))
            },
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_viewport_background(world, camera, UVec2::new(16, 12), UVec2::new(32, 40))
            },
            |_, _| {},
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
    });
}

#[test]
fn moving_a_camera_viewport_is_composite_only() {
    with_gpu_lock(|| {
        let direct = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_viewport_background(world, camera, UVec2::new(24, 12), UVec2::new(32, 40));
                camera
            },
            |_, _| {},
        );
        let moved = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_viewport_background(world, camera, UVec2::new(16, 12), UVec2::new(32, 40));
                camera
            },
            |world, camera| {
                world
                    .entity_mut(camera)
                    .get_mut::<Camera>()
                    .unwrap()
                    .viewport
                    .as_mut()
                    .unwrap()
                    .physical_position = UVec2::new(24, 12);
            },
        );

        assert_pixels_eq(&moved.pixels, &direct.pixels);
        assert_eq!(moved.paint_after_mutation, moved.paint_before_mutation);
        let before = moved.before_mutation.unwrap();
        let after = moved.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs);
        assert!(after.composites > before.composites);
    });
}

#[test]
fn resizing_a_viewport_reconstructs_its_retained_surface() {
    with_gpu_lock(|| {
        let direct = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_viewport_background(world, camera, UVec2::new(16, 12), UVec2::new(40, 40));
                camera
            },
            |_, _| {},
        );
        let resized = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_viewport_background(world, camera, UVec2::new(16, 12), UVec2::new(32, 40));
                camera
            },
            |world, camera| {
                world
                    .entity_mut(camera)
                    .get_mut::<Camera>()
                    .unwrap()
                    .viewport
                    .as_mut()
                    .unwrap()
                    .physical_size = UVec2::new(40, 40);
            },
        );

        assert_pixels_eq(&resized.pixels, &direct.pixels);
        let before = resized.before_mutation.unwrap();
        let after = resized.after_mutation.unwrap();
        assert_eq!(after.surfaces_created, before.surfaces_created + 1);
        assert_eq!(after.repairs, before.repairs + 1);
    });
}

#[test]
fn stock_image_target_persists_only_when_the_whole_camera_is_inactive() {
    with_gpu_lock(|| {
        let paint_only_skipped = render_scene(
            UiRenderer::Stock,
            PaintSchedule::UntilInitialCapture,
            |world, camera| spawn_full_background(world, camera, Color::srgb(1.0, 0.0, 0.0)),
            |_, _| {},
        );
        let camera_inactive = render_scene(
            UiRenderer::Stock,
            PaintSchedule::UntilInitialCaptureThenDisableCamera,
            |world, camera| spawn_full_background(world, camera, Color::srgb(1.0, 0.0, 0.0)),
            |_, _| {},
        );

        let center = ((HEIGHT / 2 * WIDTH + WIDTH / 2) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &paint_only_skipped.pixels[center..center + BYTES_PER_PIXEL],
            &[0, 0, 0, 0]
        );
        assert_eq!(
            &camera_inactive.pixels[center..center + BYTES_PER_PIXEL],
            &[255, 0, 0, 255]
        );
    });
}
