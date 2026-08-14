//! GPU-pixel acceptance harness for retained UI.

extern crate alloc;

use alloc::sync::Arc;
use bevy::ui_render::ui_material::{MaterialNode, UiMaterial};
use bevy::ui_render::{
    BoxShadowSamples, RenderUiSystems, UiMaterialPlugin, UiRenderInfrastructurePlugin,
    UiRenderPlugin,
};
use bevy::{
    asset::{embedded_asset, AssetId, RenderAssetUsages},
    camera::{CameraOutputMode, ClearColorConfig, RenderTarget, Viewport},
    ecs::schedule::{LogLevel, ScheduleBuildSettings, ScheduleLabel, Schedules},
    input_focus::InputFocus,
    log::LogPlugin,
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssetBytesPerFrame,
        render_resource::{
            AsBindGroup, Extent3d, PollType, TextureDimension, TextureFormat, TextureUsages,
        },
        renderer::RenderDevice,
        ExtractSchedule, Render, RenderApp, RenderPlugin,
    },
    shader::ShaderRef,
    text::{EditableText, TextCursorStyle, TextEdit, TextLayoutInfo},
    window::{ExitCondition, WindowPlugin},
};
use bevy_ui_render_retained::{
    RepaintBoundary, RetainedUiLayerCounters, RetainedUiLayerWork, RetainedUiMaterial,
    RetainedUiMaterialCoverage, RetainedUiMaterialImage, RetainedUiMaterialPlugin,
    RetainedUiMaterialSnapshot, RetainedUiPaintCounters, RetainedUiRenderPlugin, WorkCounters,
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

fn assert_no_layer_repair(before: RetainedUiLayerWork, after: RetainedUiLayerWork) {
    assert_eq!(after.repairs, before.repairs);
    assert_eq!(after.repair_pixels, before.repair_pixels);
    assert_eq!(after.items_replayed, before.items_replayed);
    assert_eq!(after.quads_replayed, before.quads_replayed);
}

fn render_scene<S>(
    renderer: UiRenderer,
    paint_schedule: PaintSchedule,
    setup: impl FnOnce(&mut World, Entity) -> S,
    mutate: impl FnOnce(&mut World, S),
) -> RenderOutput {
    render_scene_configured(renderer, paint_schedule, |_, _| {}, setup, mutate)
}

fn render_scene_configured<S>(
    renderer: UiRenderer,
    paint_schedule: PaintSchedule,
    configure: impl FnOnce(&mut App, UiRenderer),
    setup: impl FnOnce(&mut World, Entity) -> S,
    mutate: impl FnOnce(&mut World, S),
) -> RenderOutput {
    let mut app = gpu_app(renderer, paint_schedule);
    configure(&mut app, renderer);

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

fn gpu_app(renderer: UiRenderer, paint_schedule: PaintSchedule) -> App {
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
    let render_app = app.sub_app_mut(RenderApp);
    let mut schedules = render_app.world_mut().resource_mut::<Schedules>();
    for schedule in [ExtractSchedule.intern(), Render.intern()] {
        let schedule = schedules
            .get_mut(schedule)
            .expect("the render app has every retained UI schedule");
        schedule.set_build_settings(ScheduleBuildSettings {
            ambiguity_detection: LogLevel::Error,
            ..schedule.get_build_settings()
        });
    }
    app
}

#[derive(Component)]
struct FlickerProbe;

#[derive(Component)]
struct BatchedFlickerProbe;

#[derive(Component)]
struct BoundaryFlickerProbe;

#[derive(Component)]
struct BoundaryContentFlickerProbe;

fn animate_flicker_probe(
    mut probe: Single<&mut Node, With<FlickerProbe>>,
    mut position: Local<usize>,
) {
    const POSITIONS: [f32; 3] = [5.0, 21.0, 37.0];
    *position = (*position + 1) % POSITIONS.len();
    probe.left = px(POSITIONS[*position]);
}

fn batched_flicker_color(position: usize) -> Color {
    [
        Color::srgba_u8(24, 48, 96, 184),
        Color::srgba_u8(72, 32, 104, 184),
        Color::srgba_u8(28, 88, 68, 184),
    ][position]
}

fn animate_batched_flicker_probe(
    mut probe: Single<&mut BackgroundColor, With<BatchedFlickerProbe>>,
    mut position: Local<usize>,
) {
    *position = (*position + 1) % 3;
    probe.0 = batched_flicker_color(*position);
}

fn animate_boundary_flicker_probe(
    mut probe: Single<&mut RepaintBoundary, With<BoundaryFlickerProbe>>,
    mut position: Local<usize>,
) {
    const POSITIONS: [f32; 3] = [0.0, 18.0, 30.0];
    *position = (*position + 1) % POSITIONS.len();
    probe.transform = UiTransform::from_translation(Val2::px(POSITIONS[*position], 0));
}

fn animate_boundary_content_flicker_probe(
    mut probe: Single<&mut BackgroundColor, With<BoundaryContentFlickerProbe>>,
    mut position: Local<usize>,
) {
    *position = (*position + 1) % 3;
    probe.0 = batched_flicker_color(*position);
}

fn spawn_boundary_content_scene(world: &mut World, camera: Entity, position: usize) -> Entity {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    let boundary = world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(11),
                top: px(13),
                width: px(36),
                height: px(30),
                ..default()
            },
            RepaintBoundary::default(),
            ChildOf(root),
        ))
        .id();
    world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(5),
                top: px(7),
                width: px(21),
                height: px(16),
                ..default()
            },
            BackgroundColor(batched_flicker_color(position)),
            ChildOf(boundary),
        ))
        .id()
}

fn spawn_nested_visibility_scene(
    world: &mut World,
    camera: Entity,
    opacity: f32,
) -> (Entity, Entity) {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    let outer = world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(8),
                top: px(9),
                width: px(40),
                height: px(36),
                ..default()
            },
            RepaintBoundary {
                opacity,
                ..default()
            },
            ChildOf(root),
        ))
        .id();
    let inner = world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(7),
                top: px(8),
                width: px(24),
                height: px(20),
                ..default()
            },
            RepaintBoundary::default(),
            ChildOf(outer),
        ))
        .id();
    let content = world
        .spawn((
            Node {
                width: percent(100),
                height: percent(100),
                ..default()
            },
            BackgroundColor(batched_flicker_color(0)),
            ChildOf(inner),
        ))
        .id();
    (outer, content)
}

fn spawn_flicker_scene(world: &mut World, camera: Entity, left: f32) -> Entity {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(left),
                top: px(8),
                width: px(10),
                height: px(10),
                ..default()
            },
            BackgroundColor(Color::srgba_u8(220, 45, 28, 160)),
            ChildOf(root),
        ))
        .id()
}

fn spawn_batched_flicker_scene(world: &mut World, camera: Entity, position: usize) -> Entity {
    let root = spawn_full_background(world, camera, batched_flicker_color(position));
    world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(19),
            top: px(15),
            width: px(18),
            height: px(14),
            ..default()
        },
        BackgroundColor(Color::srgba_u8(230, 170, 35, 144)),
        ChildOf(root),
    ));
    root
}

fn capture_retained_stream(
    configure: impl FnOnce(&mut App),
    setup: impl FnOnce(&mut World, Entity),
) -> Vec<Vec<u8>> {
    let mut app = gpu_app(UiRenderer::Retained, PaintSchedule::EveryFrame);
    configure(&mut app);

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
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            RenderTarget::Image(image.clone().into()),
        ))
        .id();
    setup(app.world_mut(), camera);

    let frames = Arc::new(Mutex::new(Vec::<Vec<u8>>::new()));
    let observer_frames = Arc::clone(&frames);
    app.world_mut()
        .spawn(Readback::texture(image))
        .observe(move |event: On<ReadbackComplete>| {
            observer_frames
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(event.data.clone());
        });
    app.finish();
    app.cleanup();

    for _ in 0..30 {
        step_and_wait(&mut app);
    }
    frames
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clear();
    for _ in 0..32 {
        step_and_wait(&mut app);
    }

    frames
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

fn assert_complete_cycle(frames: &[Vec<u8>], references: &[Vec<u8>]) {
    assert!(
        frames.len() >= 24,
        "readback must capture the animation stream"
    );
    let states: Vec<_> = frames
        .iter()
        .map(|frame| {
            references
                .iter()
                .position(|expected| frame == expected)
                .unwrap_or_else(|| {
                    let differences: Vec<_> = references
                        .iter()
                        .map(|expected| {
                            frame
                                .iter()
                                .zip(expected)
                                .filter(|(actual, expected)| actual != expected)
                                .count()
                        })
                        .collect();
                    panic!("animation produced no complete reference state: {differences:?}");
                })
        })
        .collect();
    assert!(
        (0..references.len()).all(|state| states.contains(&state)),
        "animation omitted a complete state: {states:?}"
    );
    assert!(
        states
            .windows(2)
            .all(|pair| pair[1] == (pair[0] + 1) % references.len()),
        "each submitted animation frame must present the next complete state: {states:?}"
    );
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

fn assert_pixels_within(actual: &[u8], expected: &[u8], tolerance: u8) {
    assert_eq!(actual.len(), expected.len());
    if let Some(index) = actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual.abs_diff(*expected) > tolerance)
    {
        let pixel = index / BYTES_PER_PIXEL;
        let byte = pixel * BYTES_PER_PIXEL;
        panic!(
            "pixels first differ beyond {tolerance} at ({}, {}): actual {:?}, expected {:?}",
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

fn spawn_boundary_scene(world: &mut World, camera: Entity, retained: bool) {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    let mut boundary = world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(9),
            top: px(11),
            width: px(28),
            height: px(24),
            ..default()
        },
        BackgroundColor(Color::srgba_u8(220, 45, 28, 180)),
        ChildOf(root),
    ));
    if retained {
        boundary.insert(RepaintBoundary::default());
    }
    let boundary = boundary.id();
    world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(5),
            top: px(7),
            width: px(12),
            height: px(10),
            ..default()
        },
        BackgroundColor(Color::srgba_u8(30, 190, 90, 200)),
        ChildOf(boundary),
    ));
}

#[test]
fn identity_repaint_boundary_matches_direct_rasterization() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_boundary_scene(world, camera, false),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_boundary_scene(world, camera, true),
            |_, _| {},
        );
        assert_pixels_within(&retained.pixels, &stock.pixels, 1);
    });
}

fn spawn_ordered_boundary_scene(world: &mut World, camera: Entity, retained: bool) {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(7),
            top: px(9),
            width: px(30),
            height: px(28),
            ..default()
        },
        BackgroundColor(Color::srgba_u8(235, 180, 30, 170)),
        ChildOf(root),
    ));
    let mut boundary = world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(13),
            top: px(12),
            width: px(34),
            height: px(31),
            ..default()
        },
        BackgroundColor(Color::srgba_u8(210, 45, 35, 180)),
        ChildOf(root),
    ));
    if retained {
        boundary.insert(RepaintBoundary::default());
    }
    let boundary = boundary.id();
    world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(4),
            top: px(6),
            width: px(24),
            height: px(19),
            ..default()
        },
        BackgroundColor(Color::srgba_u8(35, 195, 90, 190)),
        ChildOf(boundary),
    ));
    world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(25),
            top: px(20),
            width: px(27),
            height: px(25),
            ..default()
        },
        BackgroundColor(Color::srgba_u8(45, 95, 225, 175)),
        ChildOf(root),
    ));
}

#[test]
fn repaint_boundary_preserves_arbitrary_sibling_paint_order() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_ordered_boundary_scene(world, camera, false),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_ordered_boundary_scene(world, camera, true),
            |_, _| {},
        );
        assert_pixels_within(&retained.pixels, &stock.pixels, 1);
    });
}

fn spawn_nested_boundary_scene(world: &mut World, camera: Entity, retained: bool) {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    let mut outer = world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(8),
            top: px(7),
            width: px(44),
            height: px(39),
            ..default()
        },
        BackgroundColor(Color::srgba_u8(220, 55, 30, 145)),
        ChildOf(root),
    ));
    if retained {
        outer.insert(RepaintBoundary::default());
    }
    let outer = outer.id();
    let mut inner = world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(9),
            top: px(8),
            width: px(26),
            height: px(23),
            ..default()
        },
        BackgroundColor(Color::srgba_u8(35, 175, 80, 165)),
        ChildOf(outer),
    ));
    if retained {
        inner.insert(RepaintBoundary::default());
    }
    let inner = inner.id();
    world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(5),
            top: px(4),
            width: px(13),
            height: px(12),
            ..default()
        },
        BackgroundColor(Color::srgba_u8(45, 80, 225, 205)),
        ChildOf(inner),
    ));
}

#[test]
fn nested_repaint_boundaries_match_direct_rasterization() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_nested_boundary_scene(world, camera, false),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_nested_boundary_scene(world, camera, true),
            |_, _| {},
        );
        assert_pixels_within(&retained.pixels, &stock.pixels, 2);
    });
}

fn spawn_clipped_boundary_scene(world: &mut World, camera: Entity, retained: bool) {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    let mut boundary = world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(17),
            top: px(15),
            width: px(24),
            height: px(20),
            overflow: Overflow::clip(),
            ..default()
        },
        BackgroundColor(Color::srgba_u8(220, 55, 30, 165)),
        ChildOf(root),
    ));
    if retained {
        boundary.insert(RepaintBoundary::default());
    }
    let boundary = boundary.id();
    world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(-7),
            top: px(6),
            width: px(39),
            height: px(9),
            ..default()
        },
        BackgroundColor(Color::srgba_u8(35, 190, 90, 210)),
        ChildOf(boundary),
    ));
}

#[test]
fn repaint_boundary_clips_its_cached_surface_exactly() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_clipped_boundary_scene(world, camera, false),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_clipped_boundary_scene(world, camera, true),
            |_, _| {},
        );
        assert_pixels_within(&retained.pixels, &stock.pixels, 1);
    });
}

#[test]
fn repaint_boundary_opacity_is_applied_after_flattening() {
    with_gpu_lock(|| {
        const RED: [f32; 4] = [0.8, 0.1, 0.05, 0.55];
        const GREEN: [f32; 4] = [0.05, 0.7, 0.2, 0.6];
        const OPACITY: f32 = 0.4;
        let source_alpha = GREEN[3] + RED[3] * (1.0 - GREEN[3]);
        let source_rgb = Vec3::new(GREEN[0], GREEN[1], GREEN[2]) * GREEN[3]
            + Vec3::new(RED[0], RED[1], RED[2]) * RED[3] * (1.0 - GREEN[3]);
        let flattened = source_rgb / source_alpha;
        let expected = Color::linear_rgba(
            flattened.x,
            flattened.y,
            flattened.z,
            source_alpha * OPACITY,
        );
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| {
                let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
                world.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(12),
                        top: px(10),
                        width: px(30),
                        height: px(26),
                        ..default()
                    },
                    BackgroundColor(expected),
                    ChildOf(root),
                ));
            },
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
                let boundary = world
                    .spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(12),
                            top: px(10),
                            width: px(30),
                            height: px(26),
                            ..default()
                        },
                        RepaintBoundary {
                            opacity: OPACITY,
                            ..default()
                        },
                        ChildOf(root),
                    ))
                    .id();
                for color in [
                    Color::linear_rgba(RED[0], RED[1], RED[2], RED[3]),
                    Color::linear_rgba(GREEN[0], GREEN[1], GREEN[2], GREEN[3]),
                ] {
                    world.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            width: percent(100),
                            height: percent(100),
                            ..default()
                        },
                        BackgroundColor(color),
                        ChildOf(boundary),
                    ));
                }
            },
            |_, _| {},
        );
        assert_pixels_within(&retained.pixels, &stock.pixels, 1);
    });
}

fn spawn_moving_boundary_scene(
    world: &mut World,
    camera: Entity,
    retained: bool,
    translation: Val2,
) -> Entity {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    let mut boundary = world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(9),
            top: px(11),
            width: px(28),
            height: px(24),
            ..default()
        },
        BackgroundColor(Color::srgba_u8(220, 45, 28, 180)),
        ChildOf(root),
    ));
    if retained {
        boundary.insert(RepaintBoundary::from_translation(translation));
    } else {
        boundary.insert(UiTransform::from_translation(translation));
    }
    let boundary = boundary.id();
    world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(5),
            top: px(7),
            width: px(12),
            height: px(10),
            ..default()
        },
        BackgroundColor(Color::srgba_u8(30, 190, 90, 200)),
        ChildOf(boundary),
    ));
    boundary
}

#[test]
fn boundary_translation_repairs_only_the_parent_surface() {
    with_gpu_lock(|| {
        let final_translation = Val2::px(18, 0);
        let direct = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_moving_boundary_scene(world, camera, false, final_translation);
            },
            |_, _| {},
        );
        let moved = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_moving_boundary_scene(world, camera, true, Val2::ZERO),
            move |world, boundary| {
                world
                    .entity_mut(boundary)
                    .get_mut::<RepaintBoundary>()
                    .unwrap()
                    .transform = UiTransform::from_translation(final_translation);
            },
        );
        assert_pixels_within(&moved.pixels, &direct.pixels, 1);

        let before = moved.before_mutation.unwrap();
        let after = moved.after_mutation.unwrap();
        assert_eq!(after.surfaces_created, before.surfaces_created);
        assert_eq!(after.surface_bytes, before.surface_bytes);
        assert_eq!(after.repairs - before.repairs, 1);
        assert_eq!(after.repair_pixels - before.repair_pixels, 46 * 24);
        assert_eq!(after.items_replayed - before.items_replayed, 2);
        assert_eq!(after.quads_replayed - before.quads_replayed, 2);

        let paint_before = moved.paint_before_mutation.unwrap();
        let paint_after = moved.paint_after_mutation.unwrap();
        assert_eq!(paint_after.candidates - paint_before.candidates, 1);
        assert_eq!(
            paint_after.records_changed - paint_before.records_changed,
            1
        );
    });
}

#[test]
fn boundary_opacity_repairs_only_the_parent_surface() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
                world
                    .spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(9),
                            top: px(11),
                            width: px(28),
                            height: px(24),
                            ..default()
                        },
                        BackgroundColor(Color::srgba_u8(220, 45, 28, 180)),
                        RepaintBoundary::default(),
                        ChildOf(root),
                    ))
                    .id()
            },
            |world, boundary| {
                world
                    .entity_mut(boundary)
                    .get_mut::<RepaintBoundary>()
                    .unwrap()
                    .opacity = 0.4;
            },
        );
        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.surfaces_created, before.surfaces_created);
        assert_eq!(after.surface_bytes, before.surface_bytes);
        assert_eq!(after.repairs - before.repairs, 1);
        assert_eq!(after.repair_pixels - before.repair_pixels, 28 * 24);
        assert_eq!(after.items_replayed - before.items_replayed, 2);
        assert_eq!(after.quads_replayed - before.quads_replayed, 2);
    });
}

#[test]
fn boundary_content_change_repairs_only_its_mapped_pixels() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                let content = spawn_boundary_content_scene(world, camera, 0);
                let boundary = world
                    .query_filtered::<Entity, With<RepaintBoundary>>()
                    .single(world)
                    .unwrap();
                world
                    .get_mut::<RepaintBoundary>(boundary)
                    .unwrap()
                    .transform = UiTransform::from_scale(Vec2::splat(2.0));
                content
            },
            |world, content| {
                world.get_mut::<BackgroundColor>(content).unwrap().0 = batched_flicker_color(1);
            },
        );
        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert_eq!(after.repairs - before.repairs, 2);
        assert_eq!(
            after.repair_pixels - before.repair_pixels,
            21 * 16 + 42 * 32
        );
        assert_eq!(after.items_replayed - before.items_replayed, 3);
        assert_eq!(after.quads_replayed - before.quads_replayed, 3);
    });
}

#[test]
fn invisible_boundary_content_changes_do_no_raster_or_parent_work() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                let content = spawn_boundary_content_scene(world, camera, 0);
                let boundary = world
                    .query_filtered::<Entity, With<RepaintBoundary>>()
                    .single(world)
                    .unwrap();
                world.get_mut::<RepaintBoundary>(boundary).unwrap().opacity = 0.0;
                content
            },
            |world, content| {
                world.get_mut::<BackgroundColor>(content).unwrap().0 = batched_flicker_color(1);
            },
        );
        assert_no_layer_repair(
            output.before_mutation.unwrap(),
            output.after_mutation.unwrap(),
        );
        let before = output.paint_before_mutation.unwrap();
        let after = output.paint_after_mutation.unwrap();
        assert_eq!(after.candidates - before.candidates, 1);
        assert_eq!(after.records_changed - before.records_changed, 1);
    });
}

#[test]
fn revealing_an_invisible_boundary_rebuilds_its_source_before_composition() {
    with_gpu_lock(|| {
        let direct = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_boundary_content_scene(world, camera, 0);
            },
            |_, _| {},
        );
        let revealed = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_boundary_content_scene(world, camera, 0);
                let boundary = world
                    .query_filtered::<Entity, With<RepaintBoundary>>()
                    .single(world)
                    .unwrap();
                world.get_mut::<RepaintBoundary>(boundary).unwrap().opacity = 0.0;
                boundary
            },
            |world, boundary| {
                world.get_mut::<RepaintBoundary>(boundary).unwrap().opacity = 1.0;
            },
        );
        assert_pixels_eq(&revealed.pixels, &direct.pixels);
        let before = revealed.before_mutation.unwrap();
        let after = revealed.after_mutation.unwrap();
        assert_eq!(after.surfaces_created - before.surfaces_created, 1);
        assert_eq!(after.surface_bytes - before.surface_bytes, 36 * 30 * 9);
        assert_eq!(after.repairs - before.repairs, 2);
        assert_eq!(after.repair_pixels - before.repair_pixels, 2 * 36 * 30);
        assert_eq!(after.items_replayed - before.items_replayed, 3);
    });
}

#[test]
fn hidden_ancestor_suppresses_nested_boundary_raster_work() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_nested_visibility_scene(world, camera, 0.0).1,
            |world, content| {
                world.get_mut::<BackgroundColor>(content).unwrap().0 = batched_flicker_color(1);
            },
        );
        assert_no_layer_repair(
            output.before_mutation.unwrap(),
            output.after_mutation.unwrap(),
        );
    });
}

#[test]
fn revealing_a_hidden_ancestor_rebuilds_nested_sources_deepest_first() {
    with_gpu_lock(|| {
        let direct = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_nested_visibility_scene(world, camera, 1.0),
            |_, _| {},
        );
        let revealed = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_nested_visibility_scene(world, camera, 0.0).0,
            |world, outer| {
                world.get_mut::<RepaintBoundary>(outer).unwrap().opacity = 1.0;
            },
        );
        assert_pixels_eq(&revealed.pixels, &direct.pixels);
        let before = revealed.before_mutation.unwrap();
        let after = revealed.after_mutation.unwrap();
        assert_eq!(after.surfaces_created - before.surfaces_created, 2);
        assert_eq!(
            after.surface_bytes - before.surface_bytes,
            (40 * 36 + 24 * 20) * 9
        );
        assert_eq!(after.repairs - before.repairs, 3);
        assert_eq!(
            after.repair_pixels - before.repair_pixels,
            40 * 36 * 2 + 24 * 20
        );
        assert_eq!(after.items_replayed - before.items_replayed, 4);
    });
}

#[test]
fn removing_a_boundary_releases_its_surface_and_preserves_pixels() {
    with_gpu_lock(|| {
        let direct = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_boundary_scene(world, camera, false),
            |_, _| {},
        );
        let removed = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_boundary_scene(world, camera, true);
                world
                    .query_filtered::<Entity, With<RepaintBoundary>>()
                    .single(world)
                    .unwrap()
            },
            |world, boundary| {
                world.entity_mut(boundary).remove::<RepaintBoundary>();
            },
        );
        assert_pixels_eq(&removed.pixels, &direct.pixels);
        let before = removed.before_mutation.unwrap();
        let after = removed.after_mutation.unwrap();
        assert_eq!(after.surfaces_created, before.surfaces_created);
        assert_eq!(before.surface_bytes - after.surface_bytes, 28 * 24 * 9);
    });
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

fn render_layout_motion(contained: bool) -> RenderOutput {
    render_scene(
        UiRenderer::Retained,
        PaintSchedule::EveryFrame,
        move |world, camera| {
            let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
            let parent = if contained {
                world
                    .spawn((
                        Node {
                            width: percent(100),
                            height: percent(100),
                            ..default()
                        },
                        LayoutContainment,
                        ChildOf(root),
                    ))
                    .id()
            } else {
                root
            };
            world
                .spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(5),
                        top: px(8),
                        width: px(10),
                        height: px(10),
                        ..default()
                    },
                    BackgroundColor(Color::srgb_u8(220, 45, 28)),
                    ChildOf(parent),
                ))
                .id()
        },
        |world, leaf| {
            world.entity_mut(leaf).get_mut::<Node>().unwrap().left = px(30);
        },
    )
}

fn assert_layout_motion(output: RenderOutput) {
    let before = output.before_mutation.unwrap();
    let after = output.after_mutation.unwrap();
    assert_eq!(after.repairs, before.repairs + 1);
    assert_eq!(after.repair_pixels, before.repair_pixels + 200);
    assert_eq!(after.items_replayed, before.items_replayed + 2);
    let paint_before = output.paint_before_mutation.unwrap();
    let paint_after = output.paint_after_mutation.unwrap();
    assert_eq!(paint_after.candidates, paint_before.candidates + 1);
    assert_eq!(
        paint_after.records_changed,
        paint_before.records_changed + 1
    );

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

fn spawn_camera_target(world: &mut World, color: [u8; 4], active: bool) -> (Entity, Handle<Image>) {
    let mut image = Image::new_fill(
        Extent3d {
            width: 20,
            height: 20,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &color,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC
        | TextureUsages::RENDER_ATTACHMENT;
    let image = world.resource_mut::<Assets<Image>>().add(image);
    let source = world
        .spawn((
            Camera2d,
            Camera {
                is_active: active,
                order: -1,
                clear_color: ClearColorConfig::Custom(Color::srgb_u8(30, 100, 220)),
                ..default()
            },
            RenderTarget::Image(image.clone().into()),
        ))
        .id();
    (source, image)
}

fn spawn_viewport_leaf(
    world: &mut World,
    camera: Entity,
    color: [u8; 4],
    source_active: bool,
) -> (Entity, Entity, Handle<Image>) {
    let (source, image) = spawn_camera_target(world, color, source_active);
    let viewport = world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(8),
                top: px(9),
                width: px(20),
                height: px(20),
                border_radius: BorderRadius::all(px(4)),
                ..default()
            },
            ViewportNode::new(source),
            UiTargetCamera(camera),
        ))
        .id();
    (viewport, source, image)
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

fn add_slice_image(world: &mut World) -> Handle<Image> {
    world.resource_mut::<Assets<Image>>().add(Image::new_fill(
        Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[
            220, 40, 30, 255, 220, 130, 20, 255, 220, 130, 20, 255, 30, 180, 70, 255, 40, 80, 210,
            255, 210, 210, 40, 255, 210, 210, 40, 255, 180, 40, 190, 255, 40, 80, 210, 255, 210,
            210, 40, 255, 210, 210, 40, 255, 180, 40, 190, 255, 40, 180, 210, 255, 130, 70, 210,
            255, 130, 70, 210, 255, 210, 90, 40, 255,
        ],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    ))
}

fn spawn_slice_leaf(world: &mut World, camera: Entity, mode: NodeImageMode, tint: Color) -> Entity {
    let image = add_slice_image(world);
    world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(7),
                top: px(9),
                width: px(24),
                height: px(20),
                ..default()
            },
            ImageNode::new(image).with_mode(mode).with_color(tint),
            UiTargetCamera(camera),
        ))
        .id()
}

fn sliced_mode() -> NodeImageMode {
    NodeImageMode::Sliced(TextureSlicer {
        border: BorderRect::all(1.0),
        center_scale_mode: SliceScaleMode::Stretch,
        sides_scale_mode: SliceScaleMode::Stretch,
        max_corner_scale: 1.0,
    })
}

fn linear_gradient(first: Color, middle: Color, last: Color) -> BackgroundGradient {
    BackgroundGradient::from(LinearGradient::to_right(vec![
        ColorStop::auto(first),
        ColorStop::percent(middle, 45),
        ColorStop::auto(last),
    ]))
}

fn spawn_gradient_leaf(world: &mut World, camera: Entity, gradient: BackgroundGradient) -> Entity {
    world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(7),
                top: px(9),
                width: px(24),
                height: px(20),
                border_radius: BorderRadius::all(px(4)),
                ..default()
            },
            gradient,
            UiTargetCamera(camera),
        ))
        .id()
}

fn spawn_shadow_leaf(world: &mut World, camera: Entity, shadow: BoxShadow) -> Entity {
    world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(16),
                top: px(14),
                width: px(20),
                height: px(20),
                border_radius: BorderRadius::all(px(5)),
                ..default()
            },
            shadow,
            UiTargetCamera(camera),
        ))
        .id()
}

fn spawn_slice_pair(world: &mut World, camera: Entity, first_tint: Color) -> Entity {
    let image = add_slice_image(world);
    let mut first = Entity::PLACEHOLDER;
    for (index, tint) in [first_tint, Color::WHITE].into_iter().enumerate() {
        let entity = world
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: px(4 + index as i32 * 28),
                    top: px(9),
                    width: px(20),
                    height: px(20),
                    ..default()
                },
                ImageNode::new(image.clone())
                    .with_mode(sliced_mode())
                    .with_color(tint),
                UiTargetCamera(camera),
            ))
            .id();
        if index == 0 {
            first = entity;
        }
    }
    first
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
fn a_camera_without_ui_uses_the_stock_final_blit_without_a_retained_surface() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |_, _| {},
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |_, _| {},
            |_, _| {},
        );
        assert_pixels_eq(&retained.pixels, &stock.pixels);

        let work = retained.after_mutation.unwrap();
        assert_eq!(work.surfaces_created, 0);
        assert_eq!(work.surface_bytes, 0);
        assert_eq!(work.repairs, 0);
        assert_eq!(work.composites, 0);
    });
}

fn spawn_stacked_camera_ui(world: &mut World, first_camera: Entity) {
    let target = world
        .get::<RenderTarget>(first_camera)
        .expect("the harness camera has an image target")
        .clone();
    spawn_full_background(world, first_camera, Color::srgb_u8(20, 80, 210));
    let second_camera = world
        .spawn((
            Camera2d,
            Camera {
                order: 1,
                clear_color: ClearColorConfig::None,
                ..default()
            },
            target,
        ))
        .id();
    spawn_full_background(world, second_camera, Color::srgba_u8(220, 40, 20, 128));
}

#[test]
fn fused_final_blit_matches_stock_multi_camera_alpha_composition() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            spawn_stacked_camera_ui,
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            spawn_stacked_camera_ui,
            |_, _| {},
        );
        assert_pixels_eq(&retained.pixels, &stock.pixels);
    });
}

#[test]
fn fused_final_writer_preserves_camera_output_skip() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera: Entity| {
            world.get_mut::<Camera>(camera).unwrap().output_mode = CameraOutputMode::Skip;
            spawn_full_background(world, camera, Color::srgb_u8(220, 40, 20))
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
        assert_eq!(retained.after_mutation.unwrap().composites, 0);
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
            "the image must produce visible pixels"
        );
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.surfaces_created, 1);
        let expected_surface_bytes =
            u64::from(WIDTH) * u64::from(HEIGHT) * (BYTES_PER_PIXEL as u64 * 2 + 1);
        assert_eq!(before.surface_bytes, expected_surface_bytes);
        assert_eq!(after.surface_bytes, expected_surface_bytes);
        assert_eq!(after.repairs, before.repairs);
        assert_eq!(
            retained.paint_after_mutation,
            retained.paint_before_mutation
        );
    });
}

#[test]
fn quiet_viewport_node_matches_stock_without_another_repair() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            spawn_viewport_leaf(world, camera, [210, 70, 25, 255], false)
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
fn modified_viewport_image_repairs_only_its_node() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            let (_, _, image) = spawn_viewport_leaf(world, camera, [210, 70, 25, 255], false);
            image
        };
        let mutate = |world: &mut World, image: Handle<Image>| {
            world
                .resource_mut::<Assets<Image>>()
                .get_mut(&image)
                .unwrap()
                .data
                .as_mut()
                .unwrap()
                .chunks_exact_mut(4)
                .for_each(|pixel| pixel.copy_from_slice(&[25, 170, 80, 255]));
        };
        let stock = render_scene(UiRenderer::Stock, PaintSchedule::EveryFrame, setup, mutate);
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            setup,
            mutate,
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 20 * 20);
        assert_eq!(after.items_replayed, before.items_replayed + 1);
        assert_eq!(after.quads_replayed, before.quads_replayed + 1);
    });
}

#[test]
fn switching_viewport_render_target_repairs_the_node() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            let (viewport, source, _) =
                spawn_viewport_leaf(world, camera, [210, 70, 25, 255], false);
            let (_, replacement) = spawn_camera_target(world, [25, 170, 80, 255], false);
            (viewport, source, replacement)
        };
        let mutate =
            |world: &mut World, (_, source, replacement): (Entity, Entity, Handle<Image>)| {
                world
                    .entity_mut(source)
                    .insert(RenderTarget::Image(replacement.into()));
            };
        let stock = render_scene(UiRenderer::Stock, PaintSchedule::EveryFrame, setup, mutate);
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            setup,
            mutate,
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 20 * 20);
    });
}

#[test]
fn active_viewport_camera_repaints_only_its_reader() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            let (_, source, _) = spawn_viewport_leaf(world, camera, [0; 4], true);
            source
        };
        let mutate = |world: &mut World, source: Entity| {
            world
                .entity_mut(source)
                .get_mut::<Camera>()
                .unwrap()
                .clear_color = ClearColorConfig::Custom(Color::srgb_u8(210, 55, 35));
        };
        let stock = render_scene(UiRenderer::Stock, PaintSchedule::EveryFrame, setup, mutate);
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            setup,
            mutate,
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        let repairs = after.repairs - before.repairs;
        assert!(repairs > 0);
        assert_eq!(
            after.repair_pixels - before.repair_pixels,
            repairs * 20 * 20
        );
    });
}

#[test]
fn skip_output_viewport_camera_is_quiet() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            let result = spawn_viewport_leaf(world, camera, [210, 70, 25, 255], true);
            world
                .entity_mut(result.1)
                .get_mut::<Camera>()
                .unwrap()
                .output_mode = CameraOutputMode::Skip;
            result
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
fn ordinary_image_node_tracks_active_camera_target_writes() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            let (source, image) = spawn_camera_target(world, [0; 4], true);
            spawn_image_leaf(world, camera, image, Color::WHITE);
            source
        };
        let mutate = |world: &mut World, source: Entity| {
            world
                .entity_mut(source)
                .get_mut::<Camera>()
                .unwrap()
                .clear_color = ClearColorConfig::Custom(Color::srgb_u8(210, 55, 35));
        };
        let stock = render_scene(UiRenderer::Stock, PaintSchedule::EveryFrame, setup, mutate);
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            setup,
            mutate,
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        let repairs = after.repairs - before.repairs;
        assert!(repairs > 0);
        assert_eq!(
            after.repair_pixels - before.repair_pixels,
            repairs * 10 * 10
        );
    });
}

#[test]
fn clearing_viewport_camera_repairs_its_vacated_pixels() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            let (viewport, _, _) = spawn_viewport_leaf(world, camera, [210, 70, 25, 255], false);
            viewport
        };
        let mutate = |world: &mut World, viewport: Entity| {
            world
                .entity_mut(viewport)
                .get_mut::<ViewportNode>()
                .unwrap()
                .camera = None;
        };
        let stock = render_scene(UiRenderer::Stock, PaintSchedule::EveryFrame, setup, mutate);
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            setup,
            mutate,
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 20 * 20);
    });
}

#[test]
fn removing_viewport_source_camera_repairs_its_vacated_pixels() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            let (_, source, _) = spawn_viewport_leaf(world, camera, [210, 70, 25, 255], false);
            source
        };
        let mutate = |world: &mut World, source: Entity| {
            world.entity_mut(source).remove::<Camera>();
        };
        let stock = render_scene(UiRenderer::Stock, PaintSchedule::EveryFrame, setup, mutate);
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            setup,
            mutate,
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 20 * 20);
    });
}

#[test]
fn equal_viewport_replacement_compares_without_repair() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                let (viewport, _, _) =
                    spawn_viewport_leaf(world, camera, [210, 70, 25, 255], false);
                viewport
            },
            |world, viewport| {
                let value = *world.entity(viewport).get::<ViewportNode>().unwrap();
                world.entity_mut(viewport).insert(value);
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

#[test]
fn quiet_sliced_image_matches_stock_without_another_repair() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            spawn_slice_leaf(world, camera, sliced_mode(), Color::WHITE)
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
fn quiet_linear_gradient_matches_stock_without_another_repair() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            spawn_gradient_leaf(
                world,
                camera,
                linear_gradient(
                    Color::srgb_u8(220, 40, 30),
                    Color::srgba_u8(30, 190, 80, 180),
                    Color::srgb_u8(40, 80, 220),
                ),
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
fn quiet_box_shadow_matches_stock_without_another_repair() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            spawn_shadow_leaf(
                world,
                camera,
                BoxShadow::new(
                    Color::srgba_u8(220, 70, 35, 210),
                    px(3),
                    px(2),
                    px(4),
                    px(3),
                ),
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
fn changed_box_shadow_repairs_its_exact_possible_pixels_and_one_quad() {
    with_gpu_lock(|| {
        let initial = Color::srgba_u8(220, 70, 35, 170);
        let final_color = Color::srgba_u8(35, 100, 230, 220);
        let setup = move |world: &mut World, camera, color| {
            spawn_shadow_leaf(
                world,
                camera,
                BoxShadow::new(color, px(3), px(2), px(4), px(2)),
            )
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, final_color),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, initial),
            move |world, entity| {
                world.entity_mut(entity).get_mut::<BoxShadow>().unwrap().0[0].color = final_color;
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 36 * 36);
        assert_eq!(after.items_replayed, before.items_replayed + 1);
        assert_eq!(after.quads_replayed, before.quads_replayed + 1);
    });
}

#[test]
fn changed_box_shadow_offset_repairs_exact_old_union_new_bounds() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera, x_offset| {
            spawn_shadow_leaf(
                world,
                camera,
                BoxShadow::new(
                    Color::srgba_u8(220, 70, 35, 210),
                    x_offset,
                    px(2),
                    px(4),
                    px(2),
                ),
            )
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, px(4)),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, px(0)),
            |world, entity| {
                world.entity_mut(entity).get_mut::<BoxShadow>().unwrap().0[0].x_offset = px(4);
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 40 * 36);
    });
}

#[test]
fn removing_box_shadow_repairs_its_vacated_pixels() {
    with_gpu_lock(|| {
        let shadow = || {
            BoxShadow::new(
                Color::srgba_u8(220, 70, 35, 210),
                px(3),
                px(2),
                px(4),
                px(2),
            )
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| {
                world
                    .spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(16),
                            top: px(14),
                            width: px(20),
                            height: px(20),
                            ..default()
                        },
                        UiTargetCamera(camera),
                    ))
                    .id()
            },
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_shadow_leaf(world, camera, shadow()),
            |world, entity| {
                world.entity_mut(entity).remove::<BoxShadow>();
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 36 * 36);
    });
}

#[test]
fn multiple_box_shadows_match_stock_back_to_front_order() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            spawn_shadow_leaf(
                world,
                camera,
                BoxShadow(vec![
                    ShadowStyle {
                        color: Color::srgba_u8(220, 40, 30, 180),
                        x_offset: px(-2),
                        y_offset: px(1),
                        spread_radius: px(5),
                        blur_radius: px(2),
                    },
                    ShadowStyle {
                        color: Color::srgba_u8(30, 90, 230, 150),
                        x_offset: px(4),
                        y_offset: px(3),
                        spread_radius: px(2),
                        blur_radius: px(1),
                    },
                ]),
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
    });
}

#[test]
fn changed_camera_shadow_samples_repairs_only_that_cameras_shadow() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera: Entity, samples: u32| {
            world.entity_mut(camera).insert(BoxShadowSamples(samples));
            (
                spawn_shadow_leaf(
                    world,
                    camera,
                    BoxShadow::new(
                        Color::srgba_u8(220, 70, 35, 210),
                        px(3),
                        px(2),
                        px(4),
                        px(5),
                    ),
                ),
                camera,
            )
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, 10),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, 1),
            |world, (_, camera)| {
                world.entity_mut(camera).insert(BoxShadowSamples(10));
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 54 * 53);
        assert_eq!(after.items_replayed, before.items_replayed + 1);
        assert_eq!(after.quads_replayed, before.quads_replayed + 1);
    });
}

#[test]
fn equal_box_shadow_replacement_compares_without_repair() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_shadow_leaf(
                    world,
                    camera,
                    BoxShadow::new(
                        Color::srgba_u8(220, 70, 35, 210),
                        px(3),
                        px(2),
                        px(4),
                        px(2),
                    ),
                )
            },
            |world, entity| {
                let shadow = world.entity(entity).get::<BoxShadow>().unwrap().clone();
                world.entity_mut(entity).insert(shadow);
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

#[test]
fn box_shadow_damage_replays_the_background_above_it() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera, color| {
            let entity = spawn_shadow_leaf(
                world,
                camera,
                BoxShadow::new(color, px(3), px(2), px(4), px(2)),
            );
            world
                .entity_mut(entity)
                .insert(BackgroundColor(Color::srgba_u8(30, 190, 80, 170)));
            entity
        };
        let final_color = Color::srgba_u8(35, 100, 230, 220);
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, final_color),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, Color::srgba_u8(220, 70, 35, 170)),
            move |world, entity| {
                world.entity_mut(entity).get_mut::<BoxShadow>().unwrap().0[0].color = final_color;
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repair_pixels, before.repair_pixels + 36 * 36);
        assert_eq!(after.items_replayed, before.items_replayed + 2);
        assert_eq!(after.quads_replayed, before.quads_replayed + 2);
    });
}

#[test]
fn transparent_box_shadow_does_no_paint_work() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                (
                    spawn_shadow_leaf(
                        world,
                        camera,
                        BoxShadow::new(Color::NONE, px(3), px(2), px(4), px(2)),
                    ),
                    camera,
                )
            },
            |world, (_, camera)| {
                world.entity_mut(camera).insert(BoxShadowSamples(10));
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
fn changed_gradient_stop_repairs_one_item_and_two_segment_quads() {
    with_gpu_lock(|| {
        let initial_middle = Color::srgba_u8(30, 190, 80, 180);
        let final_middle = Color::srgba_u8(230, 190, 25, 210);
        let setup = move |world: &mut World, camera, middle| {
            spawn_gradient_leaf(
                world,
                camera,
                linear_gradient(
                    Color::srgb_u8(220, 40, 30),
                    middle,
                    Color::srgb_u8(40, 80, 220),
                ),
            )
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, final_middle),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, initial_middle),
            move |world, entity| {
                let mut entity = world.entity_mut(entity);
                let Gradient::Linear(gradient) =
                    &mut entity.get_mut::<BackgroundGradient>().unwrap().0[0]
                else {
                    unreachable!()
                };
                gradient.stops[1].color = final_middle;
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 24 * 20);
        assert_eq!(after.items_replayed, before.items_replayed + 1);
        assert_eq!(after.quads_replayed, before.quads_replayed + 2);
    });
}

#[test]
fn single_stop_gradient_matches_stock_solid_fill() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            spawn_gradient_leaf(
                world,
                camera,
                BackgroundGradient::from(LinearGradient::to_right(vec![ColorStop::auto(
                    Color::srgba_u8(220, 60, 35, 190),
                )])),
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
    });
}

#[test]
fn changed_border_gradient_repairs_only_border_pixels() {
    with_gpu_lock(|| {
        let initial_middle = Color::srgba_u8(30, 190, 80, 180);
        let final_middle = Color::srgba_u8(230, 190, 25, 210);
        let setup = move |world: &mut World, camera, middle| {
            world
                .spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(7),
                        top: px(9),
                        width: px(24),
                        height: px(20),
                        border: UiRect::all(px(4)),
                        border_radius: BorderRadius::all(px(4)),
                        ..default()
                    },
                    BorderGradient::from(LinearGradient::to_right(vec![
                        ColorStop::auto(Color::srgb_u8(220, 40, 30)),
                        ColorStop::percent(middle, 45),
                        ColorStop::auto(Color::srgb_u8(40, 80, 220)),
                    ])),
                    UiTargetCamera(camera),
                ))
                .id()
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, final_middle),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, initial_middle),
            move |world, entity| {
                let mut entity = world.entity_mut(entity);
                let Gradient::Linear(gradient) =
                    &mut entity.get_mut::<BorderGradient>().unwrap().0[0]
                else {
                    unreachable!()
                };
                gradient.stops[1].color = final_middle;
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 340);
        assert_eq!(after.items_replayed, before.items_replayed + 1);
        // Both gradient segments carry the exact four-rectangle border damage list.
        assert_eq!(after.quads_replayed, before.quads_replayed + 2);
    });
}

#[test]
fn radial_and_conic_gradient_stack_matches_stock_order() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            spawn_gradient_leaf(
                world,
                camera,
                BackgroundGradient(vec![
                    RadialGradient::new(
                        UiPosition::CENTER,
                        RadialGradientShape::ClosestCorner,
                        vec![
                            ColorStop::auto(Color::srgba_u8(220, 40, 30, 180)),
                            ColorStop::auto(Color::srgba_u8(30, 190, 80, 80)),
                        ],
                    )
                    .into(),
                    ConicGradient::new(
                        UiPosition::CENTER,
                        vec![
                            AngularColorStop::new(Color::srgba_u8(40, 80, 220, 80), 0.0),
                            AngularColorStop::auto(Color::srgba_u8(230, 190, 25, 120)),
                            AngularColorStop::new(
                                Color::srgba_u8(220, 40, 150, 80),
                                core::f32::consts::TAU,
                            ),
                        ],
                    )
                    .with_start(0.3)
                    .into(),
                ]),
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
    });
}

#[test]
fn mixed_solid_and_multistop_gradient_stack_matches_stock_order() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            spawn_gradient_leaf(
                world,
                camera,
                BackgroundGradient(vec![
                    LinearGradient::to_right(vec![
                        ColorStop::auto(Color::srgba_u8(220, 40, 30, 160)),
                        ColorStop::auto(Color::srgba_u8(30, 190, 80, 70)),
                    ])
                    .into(),
                    LinearGradient::to_right(vec![ColorStop::auto(Color::srgba_u8(
                        40, 80, 220, 110,
                    ))])
                    .into(),
                    LinearGradient::to_top(vec![
                        ColorStop::auto(Color::srgba_u8(230, 190, 25, 90)),
                        ColorStop::auto(Color::srgba_u8(220, 40, 150, 130)),
                    ])
                    .into(),
                ]),
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
    });
}

#[test]
fn removing_background_gradient_repairs_its_vacated_pixels() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera, with_gradient| {
            let entity = world
                .spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(7),
                        top: px(9),
                        width: px(24),
                        height: px(20),
                        ..default()
                    },
                    UiTargetCamera(camera),
                ))
                .id();
            if with_gradient {
                world.entity_mut(entity).insert(linear_gradient(
                    Color::srgb_u8(220, 40, 30),
                    Color::srgba_u8(30, 190, 80, 180),
                    Color::srgb_u8(40, 80, 220),
                ));
            }
            entity
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, false),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, true),
            |world, entity| {
                world.entity_mut(entity).remove::<BackgroundGradient>();
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 24 * 20);
    });
}

#[test]
fn equal_gradient_replacement_compares_without_repair() {
    with_gpu_lock(|| {
        let output = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                spawn_gradient_leaf(
                    world,
                    camera,
                    linear_gradient(
                        Color::srgb_u8(220, 40, 30),
                        Color::srgba_u8(30, 190, 80, 180),
                        Color::srgb_u8(40, 80, 220),
                    ),
                )
            },
            |world, entity| {
                let gradient = world
                    .entity(entity)
                    .get::<BackgroundGradient>()
                    .unwrap()
                    .clone();
                world.entity_mut(entity).insert(gradient);
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

#[test]
fn sliced_image_tint_change_repairs_only_its_pixels() {
    with_gpu_lock(|| {
        let final_tint = Color::srgba_u8(70, 210, 130, 190);
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_slice_leaf(world, camera, sliced_mode(), final_tint),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_slice_leaf(world, camera, sliced_mode(), Color::WHITE),
            move |world, entity| {
                world
                    .entity_mut(entity)
                    .get_mut::<ImageNode>()
                    .unwrap()
                    .color = final_tint;
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 24 * 20);
        assert_eq!(after.items_replayed, before.items_replayed + 1);
    });
}

#[test]
fn sliced_image_repair_submits_one_quad_from_a_shared_texture() {
    with_gpu_lock(|| {
        let final_tint = Color::srgba_u8(70, 210, 130, 190);
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_slice_pair(world, camera, final_tint),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_slice_pair(world, camera, Color::WHITE),
            move |world, entity| {
                world
                    .entity_mut(entity)
                    .get_mut::<ImageNode>()
                    .unwrap()
                    .color = final_tint;
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repair_pixels, before.repair_pixels + 20 * 20);
        assert_eq!(after.items_replayed, before.items_replayed + 1);
        assert_eq!(after.quads_replayed, before.quads_replayed + 1);
    });
}

#[test]
fn switching_from_stretched_to_sliced_image_replaces_the_draw_family() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_slice_leaf(world, camera, sliced_mode(), Color::WHITE),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_slice_leaf(world, camera, NodeImageMode::Stretch, Color::WHITE),
            |world, entity| {
                world
                    .entity_mut(entity)
                    .get_mut::<ImageNode>()
                    .unwrap()
                    .image_mode = sliced_mode();
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.quads_replayed, before.quads_replayed + 1);
    });
}

#[test]
fn quiet_tiled_image_matches_stock() {
    with_gpu_lock(|| {
        let setup = |world: &mut World, camera| {
            spawn_slice_leaf(
                world,
                camera,
                NodeImageMode::Tiled {
                    tile_x: true,
                    tile_y: true,
                    stretch_value: 1.0,
                },
                Color::WHITE,
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
fn text_width_change_that_preserves_glyph_pixels_does_no_render_work() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_text_leaf(world, camera, Color::srgb_u8(220, 90, 35)),
            |world, text| {
                world.entity_mut(text).get_mut::<Node>().unwrap().width = px(57);
            },
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_text_leaf(world, camera, Color::srgb_u8(220, 90, 35)),
            |world, text| {
                world.entity_mut(text).get_mut::<Node>().unwrap().width = px(57);
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        assert_eq!(
            retained.after_mutation.unwrap().repairs,
            retained.before_mutation.unwrap().repairs
        );
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
        assert_eq!(after.items_replayed, before.items_replayed + 2);
        assert_eq!(after.quads_replayed, before.quads_replayed + 3);
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
fn switching_to_an_unavailable_image_keeps_old_pixels_without_flicker() {
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
        assert_eq!(after.repairs, before.repairs);
        assert_eq!(after.repair_pixels, before.repair_pixels);
        let center = ((14 * WIDTH + 13) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &output.pixels[center..center + BYTES_PER_PIXEL],
            &[180, 25, 70, 255]
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
fn quiet_composition_rides_the_existing_full_view_final_blit() {
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
            after.ui_sample_pixels - before.ui_sample_pixels,
            composites * u64::from(WIDTH) * u64::from(HEIGHT)
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
fn fused_composite_preserves_disjoint_pixels_and_the_gap_between_them() {
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
            after.ui_sample_pixels - before.ui_sample_pixels,
            composites * u64::from(WIDTH) * u64::from(HEIGHT)
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
fn fully_damaged_nodes_report_each_logical_item_and_quad() {
    with_gpu_lock(|| {
        let initial = Color::srgba_u8(30, 80, 180, 160);
        let final_color = Color::srgba_u8(210, 55, 40, 176);
        let setup = |world: &mut World, camera, color| {
            [px(5), px(31)].map(|left| {
                world
                    .spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left,
                            top: px(9),
                            width: px(10),
                            height: px(10),
                            ..default()
                        },
                        BackgroundColor(color),
                        UiTargetCamera(camera),
                    ))
                    .id()
            })
        };
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, final_color),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| setup(world, camera, initial),
            move |world, nodes| {
                for node in nodes {
                    world.entity_mut(node).insert(BackgroundColor(final_color));
                }
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 200);
        assert_eq!(after.items_replayed, before.items_replayed + 2);
        assert_eq!(after.quads_replayed, before.quads_replayed + 2);
    });
}

#[test]
fn moving_one_leaf_repairs_only_old_and_new_pixels_and_intersecting_items() {
    with_gpu_lock(|| {
        assert_layout_motion(render_layout_motion(false));
    });
}

#[test]
fn contained_layout_motion_repairs_the_same_exact_pixels() {
    with_gpu_lock(|| {
        assert_layout_motion(render_layout_motion(true));
    });
}

#[test]
fn repeated_repairs_never_present_an_incomplete_or_stale_layer() {
    with_gpu_lock(|| {
        let reference = |left| {
            render_scene(
                UiRenderer::Stock,
                PaintSchedule::EveryFrame,
                move |world, camera| spawn_flicker_scene(world, camera, left),
                |_, _| {},
            )
            .pixels
        };
        let references = [reference(5.0), reference(21.0), reference(37.0)];
        let frames = capture_retained_stream(
            |app| {
                app.add_systems(Update, animate_flicker_probe);
            },
            |world, camera| {
                let probe = spawn_flicker_scene(world, camera, 5.0);
                world.entity_mut(probe).insert(FlickerProbe);
            },
        );
        assert_complete_cycle(&frames, &references);
    });
}

#[test]
fn batched_full_repairs_never_present_mixed_generations() {
    with_gpu_lock(|| {
        let reference = |position| {
            render_scene(
                UiRenderer::Stock,
                PaintSchedule::EveryFrame,
                move |world, camera| spawn_batched_flicker_scene(world, camera, position),
                |_, _| {},
            )
            .pixels
        };
        let references = [reference(0), reference(1), reference(2)];
        let frames = capture_retained_stream(
            |app| {
                app.add_systems(Update, animate_batched_flicker_probe);
            },
            |world, camera| {
                let root = spawn_batched_flicker_scene(world, camera, 0);
                world.entity_mut(root).insert(BatchedFlickerProbe);
            },
        );
        assert_complete_cycle(&frames, &references);
    });
}

#[test]
fn boundary_motion_never_presents_a_stale_or_partially_repaired_frame() {
    with_gpu_lock(|| {
        let reference = |translation| {
            render_scene(
                UiRenderer::Retained,
                PaintSchedule::EveryFrame,
                move |world, camera| {
                    spawn_moving_boundary_scene(world, camera, true, Val2::px(translation, 0));
                },
                |_, _| {},
            )
            .pixels
        };
        let references = [reference(0.0), reference(18.0), reference(30.0)];
        let frames = capture_retained_stream(
            |app| {
                app.add_systems(Update, animate_boundary_flicker_probe);
            },
            |world, camera| {
                let boundary = spawn_moving_boundary_scene(world, camera, true, Val2::ZERO);
                world.entity_mut(boundary).insert(BoundaryFlickerProbe);
            },
        );
        assert_complete_cycle(&frames, &references);
    });
}

#[test]
fn boundary_content_animation_never_presents_mixed_surface_generations() {
    with_gpu_lock(|| {
        let reference = |position| {
            render_scene(
                UiRenderer::Retained,
                PaintSchedule::EveryFrame,
                move |world, camera| {
                    spawn_boundary_content_scene(world, camera, position);
                },
                |_, _| {},
            )
            .pixels
        };
        let references = [reference(0), reference(1), reference(2)];
        let frames = capture_retained_stream(
            |app| {
                app.add_systems(Update, animate_boundary_content_flicker_probe);
            },
            |world, camera| {
                let content = spawn_boundary_content_scene(world, camera, 0);
                world
                    .entity_mut(content)
                    .insert(BoundaryContentFlickerProbe);
            },
        );
        assert_complete_cycle(&frames, &references);
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
        assert_eq!(after.items_replayed, before.items_replayed + 2);
        let paint_before = output.paint_before_mutation.unwrap();
        let paint_after = output.paint_after_mutation.unwrap();
        assert_eq!(paint_after.candidates, paint_before.candidates + 1);
        assert_eq!(
            paint_after.records_changed,
            paint_before.records_changed + 1
        );
    });
}

fn spawn_moving_effects(world: &mut World, camera: Entity, left: i32) -> Entity {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(left),
                top: px(14),
                width: px(18),
                height: px(18),
                border: UiRect::all(px(3)),
                border_radius: BorderRadius::all(px(4)),
                ..default()
            },
            BackgroundColor(Color::srgba_u8(30, 80, 180, 180)),
            BorderColor::all(Color::srgba_u8(245, 220, 80, 210)),
            linear_gradient(
                Color::srgba_u8(220, 45, 28, 180),
                Color::srgba_u8(30, 190, 90, 140),
                Color::srgba_u8(35, 80, 220, 180),
            ),
            BoxShadow::new(Color::srgba_u8(0, 0, 0, 160), px(2), px(1), px(3), px(1)),
            ChildOf(root),
        ))
        .id()
}

#[test]
fn one_placement_transaction_moves_every_paint_family_without_reextracting_style() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_moving_effects(world, camera, 25),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_moving_effects(world, camera, 5),
            |world, entity| {
                world
                    .entity_mut(entity)
                    .get_mut::<UiTransform>()
                    .unwrap()
                    .translation = Val2::px(20, 0);
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.paint_before_mutation.unwrap();
        let after = retained.paint_after_mutation.unwrap();
        assert_eq!(after.candidates, before.candidates + 4);
        assert_eq!(after.records_changed, before.records_changed + 4);
    });
}

fn spawn_fully_clipped_movable_leaf(world: &mut World, camera: Entity, left: i32) -> Entity {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    let clip_parent = world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(8),
                top: px(8),
                width: px(10),
                height: px(10),
                overflow: Overflow::clip(),
                ..default()
            },
            ChildOf(root),
        ))
        .id();
    world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(left),
                top: px(0),
                width: px(10),
                height: px(10),
                ..default()
            },
            BackgroundColor(Color::srgb_u8(220, 45, 28)),
            ChildOf(clip_parent),
        ))
        .id()
}

#[test]
fn placement_can_reveal_a_fully_clipped_retained_record() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_fully_clipped_movable_leaf(world, camera, 0),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_fully_clipped_movable_leaf(world, camera, 20),
            |world, entity| {
                world
                    .entity_mut(entity)
                    .get_mut::<UiTransform>()
                    .unwrap()
                    .translation = Val2::px(-20, 0);
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.paint_before_mutation.unwrap();
        let after = retained.paint_after_mutation.unwrap();
        assert_eq!(after.candidates, before.candidates + 1);
        assert_eq!(after.records_changed, before.records_changed + 1);
    });
}

#[test]
fn placement_recovers_local_offsets_after_a_zero_scale() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            |world, camera| spawn_moving_effects(world, camera, 5),
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            |world, camera| {
                let entity = spawn_moving_effects(world, camera, 5);
                world
                    .entity_mut(entity)
                    .get_mut::<UiTransform>()
                    .unwrap()
                    .scale = Vec2::ZERO;
                entity
            },
            |world, entity| {
                world
                    .entity_mut(entity)
                    .get_mut::<UiTransform>()
                    .unwrap()
                    .scale = Vec2::ONE;
            },
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
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

fn spawn_clipped_effects(world: &mut World, camera: Entity) -> Entity {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    let clip_parent = world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(8),
                top: px(8),
                width: px(10),
                height: px(10),
                overflow: Overflow::clip(),
                ..default()
            },
            ChildOf(root),
        ))
        .id();
    world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                width: px(20),
                height: px(10),
                ..default()
            },
            linear_gradient(
                Color::srgb_u8(220, 45, 28),
                Color::srgb_u8(30, 190, 90),
                Color::srgb_u8(35, 80, 220),
            ),
            BoxShadow::new(Color::BLACK, px(4), px(0), px(2), px(0)),
            ChildOf(clip_parent),
        ))
        .id()
}

#[test]
fn retained_effect_instances_obey_ancestor_clips() {
    with_gpu_lock(|| {
        let stock = render_scene(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            spawn_clipped_effects,
            |_, _| {},
        );
        let retained = render_scene(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            spawn_clipped_effects,
            |_, _| {},
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
    });
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
        assert_eq!(
            before.surface_bytes,
            32_u64 * 40 * (BYTES_PER_PIXEL as u64 * 2 + 1)
        );
        assert_eq!(
            after.surface_bytes,
            40_u64 * 40 * (BYTES_PER_PIXEL as u64 * 2 + 1)
        );
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

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct TestUiMaterial {
    #[uniform(0)]
    color: Vec4,
    #[texture(1)]
    #[sampler(2)]
    image: Handle<Image>,
    volatile: bool,
    target_coverage: bool,
}

impl UiMaterial for TestUiMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://gpu_ui/test_ui_material.wgsl".into()
    }
}

impl RetainedUiMaterial for TestUiMaterial {
    type PaintKey = [u32; 4];

    fn retained_ui(&self) -> RetainedUiMaterialSnapshot<Self::PaintKey> {
        let coverage = if self.target_coverage {
            RetainedUiMaterialCoverage::Target
        } else {
            RetainedUiMaterialCoverage::Node
        };
        if self.volatile {
            RetainedUiMaterialSnapshot::volatile(
                coverage,
                vec![RetainedUiMaterialImage::all(self.image.id())],
            )
        } else {
            RetainedUiMaterialSnapshot::exact(
                self.color.to_array().map(f32::to_bits),
                coverage,
                vec![RetainedUiMaterialImage::all(self.image.id())],
            )
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct SecondTestUiMaterial {
    #[uniform(0)]
    color: Vec4,
}

impl UiMaterial for SecondTestUiMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://gpu_ui/test_ui_material.wgsl".into()
    }
}

impl RetainedUiMaterial for SecondTestUiMaterial {
    type PaintKey = [u32; 4];

    fn retained_ui(&self) -> RetainedUiMaterialSnapshot<Self::PaintKey> {
        RetainedUiMaterialSnapshot::exact(
            self.color.to_array().map(f32::to_bits),
            RetainedUiMaterialCoverage::Node,
            Vec::new(),
        )
    }
}

fn configure_test_ui_material(app: &mut App, renderer: UiRenderer) {
    embedded_asset!(app, "tests", "test_ui_material.wgsl");
    match renderer {
        UiRenderer::Stock => app.add_plugins(UiMaterialPlugin::<TestUiMaterial>::default()),
        UiRenderer::Retained => app.add_plugins((
            RetainedUiMaterialPlugin::<TestUiMaterial>::default(),
            RetainedUiMaterialPlugin::<SecondTestUiMaterial>::default(),
        )),
    };
}

fn configure_unretained_test_ui_material(app: &mut App, _: UiRenderer) {
    embedded_asset!(app, "tests", "test_ui_material.wgsl");
    app.add_plugins(UiMaterialPlugin::<TestUiMaterial>::default());
}

struct TestMaterialScene {
    entity: Entity,
    material: Handle<TestUiMaterial>,
    image: Handle<Image>,
}

fn spawn_test_ui_material(
    world: &mut World,
    camera: Entity,
    color: Color,
    volatile: bool,
    target_coverage: bool,
) -> TestMaterialScene {
    let image = add_solid_image(world, [255; 4]);
    let material = world
        .resource_mut::<Assets<TestUiMaterial>>()
        .add(TestUiMaterial {
            color: color.to_linear().to_vec4(),
            image: image.clone(),
            volatile,
            target_coverage,
        });
    let entity = world
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(8),
                top: px(9),
                width: px(20),
                height: px(20),
                ..default()
            },
            MaterialNode(material.clone()),
            UiTargetCamera(camera),
        ))
        .id();
    TestMaterialScene {
        entity,
        material,
        image,
    }
}

fn render_test_ui_material(
    renderer: UiRenderer,
    color: Color,
    volatile: bool,
    target_coverage: bool,
    mutate: impl FnOnce(&mut World, TestMaterialScene),
) -> RenderOutput {
    render_scene_configured(
        renderer,
        PaintSchedule::EveryFrame,
        configure_test_ui_material,
        move |world, camera| {
            spawn_test_ui_material(world, camera, color, volatile, target_coverage)
        },
        mutate,
    )
}

#[test]
fn exact_custom_material_is_quiet_after_its_first_paint() {
    with_gpu_lock(|| {
        let stock = render_test_ui_material(
            UiRenderer::Stock,
            Color::srgb_u8(220, 45, 28),
            false,
            false,
            |_, _| {},
        );
        let retained = render_test_ui_material(
            UiRenderer::Retained,
            Color::srgb_u8(220, 45, 28),
            false,
            false,
            |_, _| {},
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let sample = ((10 * WIDTH + 10) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &retained.pixels[sample..sample + BYTES_PER_PIXEL],
            &[220, 45, 28, 255]
        );
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_no_layer_repair(before, after);
        assert!(after.composites > before.composites);
        assert_eq!(
            retained.paint_after_mutation,
            retained.paint_before_mutation
        );
    });
}

#[test]
fn unretained_custom_material_uses_the_safe_full_repaint_fallback() {
    with_gpu_lock(|| {
        let output = render_scene_configured(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            configure_unretained_test_ui_material,
            |world, camera| {
                spawn_test_ui_material(world, camera, Color::srgb_u8(220, 45, 28), false, false)
            },
            |_, _| {},
        );

        let sample = ((10 * WIDTH + 10) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &output.pixels[sample..sample + BYTES_PER_PIXEL],
            &[220, 45, 28, 255],
            "layer work: before={:?}, after={:?}",
            output.before_mutation,
            output.after_mutation,
        );
        let before = output.before_mutation.unwrap();
        let after = output.after_mutation.unwrap();
        assert!(after.repairs > before.repairs);
        assert!(after.repair_pixels >= before.repair_pixels + u64::from(WIDTH * HEIGHT));
    });
}

fn spawn_unretained_material_boundary_scene(world: &mut World, camera: Entity, retained: bool) {
    let root = spawn_full_background(world, camera, Color::srgb_u8(18, 32, 76));
    let mut boundary = world.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(if retained { 8 } else { 15 }),
            top: px(if retained { 9 } else { 12 }),
            width: px(20),
            height: px(20),
            ..default()
        },
        ChildOf(root),
    ));
    if retained {
        boundary.insert(RepaintBoundary::from_translation(Val2::px(7, 3)));
    }
    let boundary = boundary.id();
    let image = add_solid_image(world, [255; 4]);
    let material = world
        .resource_mut::<Assets<TestUiMaterial>>()
        .add(TestUiMaterial {
            color: Color::srgb_u8(220, 45, 28).to_linear().to_vec4(),
            image,
            volatile: false,
            target_coverage: false,
        });
    world.spawn((
        Node {
            width: percent(100),
            height: percent(100),
            ..default()
        },
        MaterialNode(material),
        ChildOf(boundary),
    ));
}

#[test]
fn unretained_custom_material_routes_through_its_repaint_boundary() {
    with_gpu_lock(|| {
        let stock = render_scene_configured(
            UiRenderer::Stock,
            PaintSchedule::EveryFrame,
            configure_unretained_test_ui_material,
            |world, camera| spawn_unretained_material_boundary_scene(world, camera, false),
            |_, _| {},
        );
        let retained = render_scene_configured(
            UiRenderer::Retained,
            PaintSchedule::EveryFrame,
            configure_unretained_test_ui_material,
            |world, camera| spawn_unretained_material_boundary_scene(world, camera, true),
            |_, _| {},
        );

        assert_pixels_within(&retained.pixels, &stock.pixels, 1);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert!(after.repairs >= before.repairs + 2);
        assert!(after.repair_pixels >= before.repair_pixels + 800);
    });
}

#[test]
fn exact_custom_material_change_repairs_only_its_node() {
    with_gpu_lock(|| {
        let mutate = |world: &mut World, scene: TestMaterialScene| {
            world
                .resource_mut::<Assets<TestUiMaterial>>()
                .get_mut(&scene.material)
                .unwrap()
                .color = Color::srgb_u8(20, 190, 80).to_linear().to_vec4();
        };
        let stock = render_test_ui_material(
            UiRenderer::Stock,
            Color::srgb_u8(220, 45, 28),
            false,
            false,
            mutate,
        );
        let retained = render_test_ui_material(
            UiRenderer::Retained,
            Color::srgb_u8(220, 45, 28),
            false,
            false,
            mutate,
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let sample = ((10 * WIDTH + 10) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &retained.pixels[sample..sample + BYTES_PER_PIXEL],
            &[20, 190, 80, 255]
        );
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 20 * 20);
        assert_eq!(after.items_replayed, before.items_replayed + 1);
        assert_eq!(after.quads_replayed, before.quads_replayed + 1);
    });
}

#[test]
fn equal_custom_material_asset_write_does_no_paint_work() {
    with_gpu_lock(|| {
        let retained = render_test_ui_material(
            UiRenderer::Retained,
            Color::srgb_u8(220, 45, 28),
            false,
            false,
            |world, scene| {
                let mut materials = world.resource_mut::<Assets<TestUiMaterial>>();
                let color = materials.get(&scene.material).unwrap().color;
                materials.get_mut(&scene.material).unwrap().color = color;
            },
        );

        assert_no_layer_repair(
            retained.before_mutation.unwrap(),
            retained.after_mutation.unwrap(),
        );
        assert_eq!(
            retained.paint_after_mutation,
            retained.paint_before_mutation
        );
    });
}

#[test]
fn volatile_custom_material_repaints_while_visible() {
    with_gpu_lock(|| {
        let stock = render_test_ui_material(
            UiRenderer::Stock,
            Color::srgb_u8(220, 45, 28),
            true,
            false,
            |_, _| {},
        );
        let retained = render_test_ui_material(
            UiRenderer::Retained,
            Color::srgb_u8(220, 45, 28),
            true,
            false,
            |_, _| {},
        );

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert!(after.repairs > before.repairs);
        assert!(after.items_replayed > before.items_replayed);
    });
}

#[test]
fn target_coverage_custom_material_repairs_the_declared_target() {
    with_gpu_lock(|| {
        let retained = render_test_ui_material(
            UiRenderer::Retained,
            Color::srgb_u8(220, 45, 28),
            false,
            true,
            |world, scene| {
                world
                    .resource_mut::<Assets<TestUiMaterial>>()
                    .get_mut(&scene.material)
                    .unwrap()
                    .color = Color::srgb_u8(20, 190, 80).to_linear().to_vec4();
            },
        );

        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(
            after.repair_pixels,
            before.repair_pixels + u64::from(WIDTH * HEIGHT)
        );
    });
}

#[test]
fn custom_material_sample_change_repairs_only_its_reader() {
    with_gpu_lock(|| {
        let mutate = |world: &mut World, scene: TestMaterialScene| {
            world
                .resource_mut::<Assets<Image>>()
                .get_mut(&scene.image)
                .unwrap()
                .data
                .as_mut()
                .unwrap()
                .copy_from_slice(&[20, 190, 80, 255]);
        };
        let stock = render_test_ui_material(UiRenderer::Stock, Color::WHITE, false, false, mutate);
        let retained =
            render_test_ui_material(UiRenderer::Retained, Color::WHITE, false, false, mutate);

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let sample = ((10 * WIDTH + 10) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &retained.pixels[sample..sample + BYTES_PER_PIXEL],
            &[20, 190, 80, 255]
        );
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 20 * 20);
        assert_eq!(after.items_replayed, before.items_replayed + 1);
    });
}

#[test]
fn custom_material_waits_for_a_changed_image_binding() {
    with_gpu_lock(|| {
        let mutate = |world: &mut World, scene: TestMaterialScene| {
            let image = add_solid_image(world, [30, 70, 210, 255]);
            world
                .resource_mut::<Assets<TestUiMaterial>>()
                .get_mut(&scene.material)
                .unwrap()
                .image = image;
        };
        let stock = render_test_ui_material(UiRenderer::Stock, Color::WHITE, false, false, mutate);
        let retained =
            render_test_ui_material(UiRenderer::Retained, Color::WHITE, false, false, mutate);

        assert_pixels_eq(&retained.pixels, &stock.pixels);
        let sample = ((10 * WIDTH + 10) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &retained.pixels[sample..sample + BYTES_PER_PIXEL],
            &[30, 70, 210, 255]
        );
        let before = retained.before_mutation.unwrap();
        let after = retained.after_mutation.unwrap();
        assert_eq!(after.repairs, before.repairs + 1);
        assert_eq!(after.repair_pixels, before.repair_pixels + 20 * 20);
    });
}

#[test]
fn unprepared_custom_material_binding_keeps_old_pixels_and_damage_owed() {
    with_gpu_lock(|| {
        let retained = render_test_ui_material(
            UiRenderer::Retained,
            Color::WHITE,
            false,
            false,
            |world, scene| {
                world.insert_resource(RenderAssetBytesPerFrame::new(0));
                let image = add_solid_image(world, [30, 70, 210, 255]);
                world
                    .resource_mut::<Assets<TestUiMaterial>>()
                    .get_mut(&scene.material)
                    .unwrap()
                    .image = image;
            },
        );

        let sample = ((10 * WIDTH + 10) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &retained.pixels[sample..sample + BYTES_PER_PIXEL],
            &[255, 255, 255, 255]
        );
        assert_no_layer_repair(
            retained.before_mutation.unwrap(),
            retained.after_mutation.unwrap(),
        );
        let before = retained.paint_before_mutation.unwrap();
        let after = retained.paint_after_mutation.unwrap();
        assert_eq!(after.records_changed, before.records_changed + 1);
    });
}

#[test]
fn removing_a_custom_material_repairs_its_previous_pixels() {
    with_gpu_lock(|| {
        let retained = render_test_ui_material(
            UiRenderer::Retained,
            Color::srgb_u8(220, 45, 28),
            false,
            false,
            |world, scene| {
                world
                    .entity_mut(scene.entity)
                    .remove::<MaterialNode<TestUiMaterial>>();
            },
        );

        let sample = ((10 * WIDTH + 10) as usize) * BYTES_PER_PIXEL;
        assert_eq!(
            &retained.pixels[sample..sample + BYTES_PER_PIXEL],
            &[0, 0, 0, 255]
        );
        let before = retained.paint_before_mutation.unwrap();
        let after = retained.paint_after_mutation.unwrap();
        assert_eq!(after.records_removed, before.records_removed + 1);
    });
}

/// A record whose coverage is EMPTY across an ordering rebuild must not
/// keep its group index from the PREVIOUS ordering. Regression:
/// `owned.group` was only reassigned for records included in the rebuild
/// (non-empty coverage), and a TEXT record persists with empty coverage
/// when its string empties — so a readout cleared to "" while the
/// ordering shrank came back holding a group index past the rebuilt
/// per-group vectors, and `note_direct` panicked with `index out of
/// bounds` in `direct_epochs` (observed live from
/// `extract_retained_text`, len 273 / index 274).
#[test]
fn text_emptied_across_a_shrinking_reorder_refills_safely() {
    with_gpu_lock(|| {
        let mut app = gpu_app(UiRenderer::Retained, PaintSchedule::EveryFrame);

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
                    clear_color: ClearColorConfig::Custom(Color::BLACK),
                    ..default()
                },
                RenderTarget::Image(image.clone().into()),
            ))
            .id();

        // A population whose ordering has several groups; the victim
        // spawns LAST so it lands in the highest group.
        let world = app.world_mut();
        let root = spawn_full_background(world, camera, Color::srgb_u8(10, 10, 30));
        let mut crowd = Vec::new();
        for i in 0..6 {
            crowd.push(
                world
                    .spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(2 + 8 * i),
                            top: px(4),
                            width: px(6),
                            height: px(6),
                            ..default()
                        },
                        BackgroundColor(Color::srgb_u8(40 + 20 * i as u8, 120, 60)),
                        ChildOf(root),
                    ))
                    .id(),
            );
        }
        let victim = world
            .spawn((
                Text::new("live readout"),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(2),
                    top: px(30),
                    ..default()
                },
                ChildOf(root),
            ))
            .id();

        app.finish();
        app.cleanup();
        for _ in 0..20 {
            step_and_wait(&mut app);
        }

        // Empty the readout (empty coverage, record persists, EXCLUDED
        // from the next ordering) and shrink the ordering under it.
        let world = app.world_mut();
        *world.get_mut::<Text>(victim).unwrap() = Text::new("");
        for node in crowd {
            world.entity_mut(node).despawn();
        }
        for _ in 0..8 {
            step_and_wait(&mut app);
        }

        // The readout refills. Its old group index now points past the
        // rebuilt (smaller) per-group vectors.
        *app.world_mut().get_mut::<Text>(victim).unwrap() = Text::new("back");
        for _ in 0..8 {
            step_and_wait(&mut app);
        }
    });
}

/// A `UiFillsTarget` camera's interface covers the whole render target,
/// not the camera's viewport: layout sizes to the target, and the layer
/// composites across the full output. Without the marker, the same node
/// lays out inside the viewport band. Both halves are asserted so the
/// marker is proven to be the discriminator.
#[test]
fn a_fills_target_camera_lays_its_interface_over_the_whole_target() {
    with_gpu_lock(|| {
        for fills in [false, true] {
            let mut app = gpu_app(UiRenderer::Retained, PaintSchedule::EveryFrame);

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
            // The camera renders only the TOP HALF of the target.
            let mut camera = app.world_mut().spawn((
                Camera2d,
                Camera {
                    clear_color: ClearColorConfig::Custom(Color::BLACK),
                    viewport: Some(Viewport {
                        physical_position: UVec2::ZERO,
                        physical_size: UVec2::new(WIDTH, HEIGHT / 2),
                        depth: 0.0..1.0,
                    }),
                    ..default()
                },
                RenderTarget::Image(image.clone().into()),
            ));
            if fills {
                camera.insert(bevy::ui::UiFillsTarget);
            }
            let camera = camera.id();

            // A node in the BOTTOM half of the TARGET — outside the
            // viewport, inside the target.
            app.world_mut().spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: px(8),
                    top: px((HEIGHT * 3 / 4) as i32),
                    width: px(16),
                    height: px(10),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.0, 1.0, 0.0)),
                bevy::ui::UiTargetCamera(camera),
            ));

            let pixels = Arc::new(Mutex::new(None));
            let observer_pixels = Arc::clone(&pixels);
            app.world_mut().spawn(Readback::texture(image)).observe(
                move |event: On<ReadbackComplete>| {
                    *observer_pixels
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner) = Some(event.data.clone());
                },
            );
            app.finish();
            app.cleanup();
            for _ in 0..20 {
                step_and_wait(&mut app);
            }
            let frame = capture_fresh(&mut app, &pixels);
            // Sample inside the node's target-space rect.
            let x = 12u32;
            let y = HEIGHT * 3 / 4 + 4;
            let i = ((y * WIDTH + x) * BYTES_PER_PIXEL as u32) as usize;
            let px_val = &frame[i..i + 3];
            if fills {
                assert!(
                    px_val[1] > 200,
                    "fills-target: the node should paint at its TARGET position \
                     below the viewport (got {px_val:?})"
                );
            } else {
                assert!(
                    px_val[1] < 50,
                    "without the marker the interface is viewport-bound; green \
                     below the viewport means the default changed (got {px_val:?})"
                );
            }
        }
    });
}

/// A repaint boundary that MOVES (layout reflow after a sibling despawns)
/// must reposition its cached surface without billing any repair: the
/// survivors' content is byte-identical, only placement changed. This is
/// the contract a toast rack leans on — an expiring toast reflows the
/// stack every few seconds, and if reposition billed paint, every reflow
/// would re-shade whatever heavy materials sit under the rack.
#[test]
#[ignore = "FORK GAP: boundary placement is cached-content but not \
compositor-free — a layout reflow repositions correctly (both pixel \
asserts pass) yet bills one cached-blit repair per survivor in the parent \
layer. Remove this ignore for the red repro of true placement independence."]
fn a_boundary_reflow_repositions_without_repair() {
    with_gpu_lock(|| {
        let mut app = gpu_app(UiRenderer::Retained, PaintSchedule::EveryFrame);

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
                    clear_color: ClearColorConfig::Custom(Color::BLACK),
                    ..default()
                },
                RenderTarget::Image(image.clone().into()),
            ))
            .id();

        // A column rack of three boundary "toasts", distinct colors.
        let world = app.world_mut();
        let rack = world
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: px(4),
                    top: px(4),
                    flex_direction: FlexDirection::Column,
                    row_gap: px(2),
                    ..default()
                },
                bevy::ui::UiTargetCamera(camera),
            ))
            .id();
        let mut toasts = Vec::new();
        for (r, g, b) in [(255, 0, 0), (0, 255, 0), (0, 0, 255)] {
            toasts.push(
                world
                    .spawn((
                        RepaintBoundary::IDENTITY,
                        Node {
                            width: px(24),
                            height: px(10),
                            ..default()
                        },
                        BackgroundColor(Color::srgb_u8(r, g, b)),
                        ChildOf(rack),
                    ))
                    .id(),
            );
        }

        let pixels = Arc::new(Mutex::new(None));
        let observer_pixels = Arc::clone(&pixels);
        app.world_mut().spawn(Readback::texture(image)).observe(
            move |event: On<ReadbackComplete>| {
                *observer_pixels
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner) = Some(event.data.clone());
            },
        );
        app.finish();
        app.cleanup();
        for _ in 0..20 {
            step_and_wait(&mut app);
        }
        let frame = capture_fresh(&mut app, &pixels);
        // Baseline: red at row 0, green at row 1 (rows at y=4.. and y=16..).
        let at = |f: &Vec<u8>, x: u32, y: u32| {
            let i = ((y * WIDTH + x) * BYTES_PER_PIXEL as u32) as usize;
            [f[i], f[i + 1], f[i + 2]]
        };
        assert_eq!(at(&frame, 8, 8), [255, 0, 0], "baseline row 0 is red");
        assert_eq!(at(&frame, 8, 20), [0, 255, 0], "baseline row 1 is green");
        let before = layer_work(&app).expect("layer counters");

        // The oldest toast expires; the rack reflows.
        app.world_mut().entity_mut(toasts[0]).despawn();
        for _ in 0..6 {
            step_and_wait(&mut app);
        }
        let frame = capture_fresh(&mut app, &pixels);
        let after = layer_work(&app).expect("layer counters");

        // The survivors moved up (green now in row-0 position)...
        assert_eq!(
            at(&frame, 8, 8),
            [0, 255, 0],
            "after the reflow the green toast should occupy the first slot"
        );
        assert_eq!(at(&frame, 8, 20), [0, 0, 255], "and blue the second");
        // ...and the move billed NO repair: cached surfaces recomposited
        // at new offsets, no pixels repainted.
        assert_eq!(
            after.repair_pixels, before.repair_pixels,
            "a boundary reflow must not repaint pixels (repairs {} -> {}, \
             repair_pixels {} -> {})",
            before.repairs, after.repairs, before.repair_pixels, after.repair_pixels
        );
    });
}

/// Animating a boundary's OWN transform (the compositor-placement lane)
/// must bill zero repair across every frame of the motion — this is the
/// slide-in a toast rides, and the whole point of driving it through the
/// boundary instead of layout.
#[test]
#[ignore = "FORK GAP: animating RepaintBoundary.transform repositions \
correctly but bills a quad-area cached-blit repair in the parent layer \
EVERY moved frame (repairs +1/frame). The blit is cheap (no content \
shaders re-run) so islands still pay off — but placement should cost \
zero. Remove this ignore for the red repro."]
fn a_boundary_transform_slide_bills_no_repair() {
    with_gpu_lock(|| {
        let mut app = gpu_app(UiRenderer::Retained, PaintSchedule::EveryFrame);

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
                    clear_color: ClearColorConfig::Custom(Color::BLACK),
                    ..default()
                },
                RenderTarget::Image(image.clone().into()),
            ))
            .id();
        let world = app.world_mut();
        let toast = world
            .spawn((
                RepaintBoundary::IDENTITY,
                Node {
                    position_type: PositionType::Absolute,
                    left: px(4),
                    top: px(4),
                    width: px(20),
                    height: px(10),
                    ..default()
                },
                BackgroundColor(Color::srgb_u8(255, 0, 0)),
                bevy::ui::UiTargetCamera(camera),
            ))
            .id();

        let pixels = Arc::new(Mutex::new(None));
        let observer_pixels = Arc::clone(&pixels);
        app.world_mut().spawn(Readback::texture(image)).observe(
            move |event: On<ReadbackComplete>| {
                *observer_pixels
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner) = Some(event.data.clone());
            },
        );
        app.finish();
        app.cleanup();
        for _ in 0..20 {
            step_and_wait(&mut app);
        }
        capture_fresh(&mut app, &pixels);
        let before = layer_work(&app).expect("layer counters");

        // Slide 24px right, 2px per frame — every frame is a move.
        for step in 1..=12 {
            app.world_mut()
                .entity_mut(toast)
                .get_mut::<RepaintBoundary>()
                .unwrap()
                .transform =
                bevy::ui::UiTransform::from_translation(bevy::ui::Val2::px(step as f32 * 2.0, 0.0));
            step_and_wait(&mut app);
        }
        let frame = capture_fresh(&mut app, &pixels);
        let after = layer_work(&app).expect("layer counters");

        let at = |f: &Vec<u8>, x: u32, y: u32| {
            let i = ((y * WIDTH + x) * BYTES_PER_PIXEL as u32) as usize;
            [f[i], f[i + 1], f[i + 2]]
        };
        // It moved: the old spot is bare, the slid-to spot is red.
        assert_eq!(at(&frame, 6, 8), [0, 0, 0], "the origin should be vacated");
        assert_eq!(
            at(&frame, 30, 8),
            [255, 0, 0],
            "the toast should sit 24px right of where it started"
        );
        assert_eq!(
            after.repair_pixels, before.repair_pixels,
            "a transform slide must not repaint pixels (repairs {} -> {}, \
             repair_pixels {} -> {})",
            before.repairs, after.repairs, before.repair_pixels, after.repair_pixels
        );
    });
}
