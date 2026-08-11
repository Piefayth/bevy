//! Windowed retained-UI stress matrix for desktop and physical-device measurements.

#![expect(
    clippy::print_stdout,
    reason = "the command-line stress tool prints help and its exact configuration"
)]

use bevy::{
    asset::RenderAssetUsages,
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        Render, RenderApp,
    },
    ui_render::{UiRenderInfrastructurePlugin, UiRenderPlugin},
    window::{PresentMode, WindowResolution},
    winit::WinitSettings,
};
use bevy_ui_render_retained::{
    RetainedUiLayerCounters, RetainedUiMainWorldCounters, RetainedUiPaintCounters,
    RetainedUiRenderPlugin,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Renderer {
    Stock,
    Retained,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Geometry {
    Grid,
    Overlap,
    Alternating,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Background,
    Text,
    Image,
    Effects,
    Mixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ItemFamily {
    Background,
    Text,
    Image,
    Effects,
}

impl Family {
    const fn item(self, index: usize) -> ItemFamily {
        match self {
            Self::Background => ItemFamily::Background,
            Self::Text => ItemFamily::Text,
            Self::Image => ItemFamily::Image,
            Self::Effects => ItemFamily::Effects,
            Self::Mixed => match index % 4 {
                0 => ItemFamily::Background,
                1 => ItemFamily::Text,
                2 => ItemFamily::Image,
                _ => ItemFamily::Effects,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Workload {
    Quiet,
    OnePaint,
    AllPaint,
    OnePlacement,
    AllPlacement,
    OneLayout,
    AllLayout,
    OneChurn,
}

#[derive(Resource, Clone, Debug)]
struct Config {
    renderer: Renderer,
    geometry: Geometry,
    family: Family,
    workload: Workload,
    nodes: usize,
    frames: Option<u32>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            renderer: Renderer::Retained,
            geometry: Geometry::Grid,
            family: Family::Background,
            workload: Workload::Quiet,
            nodes: 10_000,
            frames: None,
        }
    }
}

impl Config {
    fn parse() -> Self {
        let mut config = Self::default();
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            let mut value = || {
                arguments
                    .next()
                    .unwrap_or_else(|| panic!("{argument} requires a value"))
            };
            match argument.as_str() {
                "--renderer" => {
                    config.renderer = match value().as_str() {
                        "stock" => Renderer::Stock,
                        "retained" => Renderer::Retained,
                        value => panic!("unknown renderer {value:?}"),
                    };
                }
                "--geometry" => {
                    config.geometry = match value().as_str() {
                        "grid" => Geometry::Grid,
                        "overlap" => Geometry::Overlap,
                        "alternating" => Geometry::Alternating,
                        value => panic!("unknown geometry {value:?}"),
                    };
                }
                "--family" => {
                    config.family = match value().as_str() {
                        "background" => Family::Background,
                        "text" => Family::Text,
                        "image" => Family::Image,
                        "effects" => Family::Effects,
                        "mixed" => Family::Mixed,
                        value => panic!("unknown family {value:?}"),
                    };
                }
                "--workload" => {
                    config.workload = match value().as_str() {
                        "quiet" => Workload::Quiet,
                        "one-paint" => Workload::OnePaint,
                        "all-paint" => Workload::AllPaint,
                        "one-placement" => Workload::OnePlacement,
                        "all-placement" => Workload::AllPlacement,
                        "one-layout" => Workload::OneLayout,
                        "all-layout" => Workload::AllLayout,
                        "one-churn" => Workload::OneChurn,
                        value => panic!("unknown workload {value:?}"),
                    };
                }
                "--nodes" => config.nodes = value().parse().expect("--nodes must be an integer"),
                "--frames" => {
                    config.frames = Some(value().parse().expect("--frames must be an integer"));
                }
                "--help" => {
                    println!(
                        "--renderer stock|retained --geometry grid|overlap|alternating \
                         --family background|text|image|effects|mixed \
                         --workload quiet|one-paint|all-paint|one-placement|all-placement|\
                         one-layout|all-layout|one-churn --nodes N [--frames N]"
                    );
                    std::process::exit(0);
                }
                value => panic!("unknown argument {value:?}; use --help"),
            }
        }
        assert!(config.nodes > 0, "--nodes must be nonzero");
        config
    }
}

#[derive(Resource)]
struct StressNodes {
    root: Entity,
    nodes: Vec<StressItem>,
    image: Handle<Image>,
    alternate: bool,
}

#[derive(Clone, Copy)]
struct StressItem {
    entity: Entity,
    family: ItemFamily,
}

fn main() {
    let config = Config::parse();
    println!("retained-ui-stress {config:?}");

    let mut plugins = DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: format!(
                "retained-ui {:?} {:?} {:?} {:?} {}",
                config.renderer, config.geometry, config.family, config.workload, config.nodes
            ),
            present_mode: PresentMode::AutoNoVsync,
            resolution: WindowResolution::new(1280, 720).with_scale_factor_override(1.0),
            ..Default::default()
        }),
        ..Default::default()
    });
    if config.renderer == Renderer::Retained {
        plugins = plugins.disable::<UiRenderPlugin>();
    }

    let mut app = App::new();
    app.add_plugins(plugins)
        .add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            LogDiagnosticsPlugin::default(),
        ))
        .insert_resource(WinitSettings::continuous())
        .insert_resource(config.clone())
        .add_systems(Startup, setup);

    if config.renderer == Renderer::Retained {
        app.add_plugins((UiRenderInfrastructurePlugin, RetainedUiRenderPlugin));
        app.add_systems(Update, report_main_work);
        app.sub_app_mut(RenderApp)
            .add_systems(Render, report_render_work);
    }

    match config.workload {
        Workload::Quiet => {}
        Workload::OnePaint => {
            app.add_systems(Update, animate_one_paint);
        }
        Workload::AllPaint => {
            app.add_systems(Update, animate_all_paint);
        }
        Workload::OnePlacement => {
            app.add_systems(Update, animate_one_placement);
        }
        Workload::AllPlacement => {
            app.add_systems(Update, animate_all_placement);
        }
        Workload::OneLayout => {
            app.add_systems(Update, animate_one_layout);
        }
        Workload::AllLayout => {
            app.add_systems(Update, animate_all_layout);
        }
        Workload::OneChurn => {
            app.add_systems(Update, churn_one);
        }
    };
    if config.frames.is_some() {
        app.add_systems(Update, exit_after_frames);
    }
    app.run();
}

fn setup(mut commands: Commands, config: Res<Config>, mut images: ResMut<Assets<Image>>) {
    commands.spawn(Camera2d);
    let image = images.add(Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[70, 160, 230, 128],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    ));
    let root = commands
        .spawn(Node {
            width: percent(100),
            height: percent(100),
            ..Default::default()
        })
        .id();
    let columns = (config.nodes as f32).sqrt().ceil() as usize;
    let mut nodes = Vec::with_capacity(config.nodes);
    for index in 0..config.nodes {
        let family = config.family.item(index);
        let entity = spawn_item(
            &mut commands,
            root,
            family,
            config.geometry,
            index,
            columns,
            &image,
        );
        nodes.push(StressItem { entity, family });
    }
    commands.insert_resource(StressNodes {
        root,
        nodes,
        image,
        alternate: false,
    });
}

fn spawn_item(
    commands: &mut Commands,
    root: Entity,
    family: ItemFamily,
    geometry: Geometry,
    index: usize,
    columns: usize,
    image: &Handle<Image>,
) -> Entity {
    let (left, top, size) = match geometry {
        Geometry::Grid => (
            (index % columns) as f32 * 7.0,
            (index / columns) as f32 * 7.0,
            6.0,
        ),
        Geometry::Overlap => (100.0, 100.0, 64.0),
        Geometry::Alternating if index.is_multiple_of(2) => (100.0, 100.0, 64.0),
        Geometry::Alternating => (300.0, 100.0, 64.0),
    };
    let mut entity = commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(left),
            top: px(top),
            width: px(size),
            height: px(size),
            border: if family == ItemFamily::Effects {
                UiRect::all(px(1.0))
            } else {
                UiRect::ZERO
            },
            ..Default::default()
        },
        ChildOf(root),
    ));
    match family {
        ItemFamily::Background => {
            entity.insert(BackgroundColor(colors(false)));
        }
        ItemFamily::Text => {
            entity.insert((
                Text::new("A"),
                TextFont {
                    font_size: FontSize::Px(size),
                    ..Default::default()
                },
                TextColor(colors(false)),
            ));
        }
        ItemFamily::Image => {
            let mut node = ImageNode::new(image.clone());
            node.color = colors(false);
            entity.insert(node);
        }
        ItemFamily::Effects => {
            entity.insert((
                BackgroundColor(colors(false)),
                BorderColor::all(Color::srgba(0.9, 0.8, 0.2, 0.7)),
                BackgroundGradient::from(LinearGradient::to_right(vec![
                    ColorStop::auto(Color::srgba(0.2, 0.8, 0.4, 0.6)),
                    ColorStop::auto(Color::srgba(0.7, 0.2, 0.8, 0.6)),
                ])),
                BoxShadow::new(
                    Color::srgba(0.0, 0.0, 0.0, 0.5),
                    px(2.0),
                    px(2.0),
                    px(1.0),
                    px(2.0),
                ),
            ));
        }
    }
    entity.id()
}

fn colors(alternate: bool) -> Color {
    if alternate {
        Color::srgba(0.8, 0.3, 0.2, 0.5)
    } else {
        Color::srgba(0.2, 0.5, 0.8, 0.5)
    }
}

fn set_paint(
    item: StressItem,
    color: Color,
    backgrounds: &mut Query<&mut BackgroundColor>,
    texts: &mut Query<&mut TextColor>,
    images: &mut Query<&mut ImageNode>,
) {
    match item.family {
        ItemFamily::Background | ItemFamily::Effects => {
            backgrounds.get_mut(item.entity).unwrap().0 = color;
        }
        ItemFamily::Text => texts.get_mut(item.entity).unwrap().0 = color,
        ItemFamily::Image => images.get_mut(item.entity).unwrap().color = color,
    }
}

fn animate_one_paint(
    mut stress: ResMut<StressNodes>,
    mut backgrounds: Query<&mut BackgroundColor>,
    mut texts: Query<&mut TextColor>,
    mut images: Query<&mut ImageNode>,
) {
    stress.alternate = !stress.alternate;
    set_paint(
        stress.nodes[0],
        colors(stress.alternate),
        &mut backgrounds,
        &mut texts,
        &mut images,
    );
}

fn animate_all_paint(
    mut stress: ResMut<StressNodes>,
    mut backgrounds: Query<&mut BackgroundColor>,
    mut texts: Query<&mut TextColor>,
    mut images: Query<&mut ImageNode>,
) {
    stress.alternate = !stress.alternate;
    let color = colors(stress.alternate);
    for &item in &stress.nodes {
        set_paint(item, color, &mut backgrounds, &mut texts, &mut images);
    }
}

fn animate_one_placement(mut stress: ResMut<StressNodes>, mut transforms: Query<&mut UiTransform>) {
    stress.alternate = !stress.alternate;
    transforms
        .get_mut(stress.nodes[0].entity)
        .unwrap()
        .translation
        .x = px(if stress.alternate { 1.0 } else { 0.0 });
}

fn animate_all_placement(mut stress: ResMut<StressNodes>, mut transforms: Query<&mut UiTransform>) {
    stress.alternate = !stress.alternate;
    let x = px(if stress.alternate { 1.0 } else { 0.0 });
    for item in &stress.nodes {
        transforms.get_mut(item.entity).unwrap().translation.x = x;
    }
}

fn animate_one_layout(mut stress: ResMut<StressNodes>, mut nodes: Query<&mut Node>) {
    stress.alternate = !stress.alternate;
    nodes.get_mut(stress.nodes[0].entity).unwrap().width =
        px(if stress.alternate { 7.0 } else { 6.0 });
}

fn animate_all_layout(mut stress: ResMut<StressNodes>, mut nodes: Query<&mut Node>) {
    stress.alternate = !stress.alternate;
    let width = px(if stress.alternate { 7.0 } else { 6.0 });
    for item in &stress.nodes {
        nodes.get_mut(item.entity).unwrap().width = width;
    }
}

fn churn_one(mut commands: Commands, config: Res<Config>, mut stress: ResMut<StressNodes>) {
    let old = stress.nodes[0];
    commands.entity(old.entity).despawn();
    let entity = spawn_item(
        &mut commands,
        stress.root,
        old.family,
        config.geometry,
        0,
        (config.nodes as f32).sqrt().ceil() as usize,
        &stress.image,
    );
    stress.nodes[0].entity = entity;
}

fn report_main_work(main: Res<RetainedUiMainWorldCounters>, mut frames: Local<u32>) {
    *frames += 1;
    if (*frames).is_multiple_of(120) {
        info!("retained main={:?}", main.snapshot());
    }
}

fn report_render_work(
    paint: Res<RetainedUiPaintCounters>,
    layer: Res<RetainedUiLayerCounters>,
    mut frames: Local<u32>,
) {
    *frames += 1;
    if (*frames).is_multiple_of(120) {
        info!(
            "retained paint={:?} layer={:?}",
            paint.snapshot(),
            layer.snapshot()
        );
    }
}

fn exit_after_frames(
    config: Res<Config>,
    mut frames: Local<u32>,
    mut exit: MessageWriter<AppExit>,
) {
    *frames += 1;
    if config.frames.is_some_and(|limit| *frames >= limit) {
        exit.write(AppExit::Success);
    }
}
