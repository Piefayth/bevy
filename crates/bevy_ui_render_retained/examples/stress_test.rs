//! Windowed retained-UI stress matrix for desktop and physical-device measurements.

#![expect(
    clippy::print_stdout,
    reason = "the command-line stress tool prints help and its exact configuration"
)]

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        Render, RenderApp,
    },
    time::Real,
    ui_render::{UiRenderInfrastructurePlugin, UiRenderPlugin},
    window::{PresentMode, WindowResolution},
    winit::WinitSettings,
};
use bevy_ui_render_retained::{
    RepaintBoundary, RetainedUiLayerCounters, RetainedUiMainWorldCounters, RetainedUiPaintCounters,
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
    OneBoundaryPlacement,
    AllBoundaryPlacement,
    OneChurn,
}

#[derive(Resource, Clone, Debug)]
struct Config {
    renderer: Renderer,
    geometry: Geometry,
    family: Family,
    workload: Workload,
    nodes: usize,
    layout_group: Option<usize>,
    frames: Option<u32>,
    warmup_frames: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            renderer: Renderer::Retained,
            geometry: Geometry::Grid,
            family: Family::Background,
            workload: Workload::Quiet,
            nodes: 10_000,
            layout_group: None,
            frames: None,
            warmup_frames: 120,
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
                        "one-boundary-placement" => Workload::OneBoundaryPlacement,
                        "all-boundary-placement" => Workload::AllBoundaryPlacement,
                        "one-churn" => Workload::OneChurn,
                        value => panic!("unknown workload {value:?}"),
                    };
                }
                "--nodes" => config.nodes = value().parse().expect("--nodes must be an integer"),
                "--layout-group" => {
                    config.layout_group =
                        Some(value().parse().expect("--layout-group must be an integer"));
                }
                "--frames" => {
                    config.frames = Some(value().parse().expect("--frames must be an integer"));
                }
                "--warmup" => {
                    config.warmup_frames = value().parse().expect("--warmup must be an integer");
                }
                "--help" => {
                    println!(
                        "--renderer stock|retained --geometry grid|overlap|alternating \
                         --family background|text|image|effects|mixed \
                         --workload quiet|one-paint|all-paint|one-placement|all-placement|\
                         one-layout|all-layout|one-boundary-placement|all-boundary-placement|\
                         one-churn --nodes N [--layout-group N] [--frames N] \
                         [--warmup N]"
                    );
                    std::process::exit(0);
                }
                value => panic!("unknown argument {value:?}; use --help"),
            }
        }
        assert!(config.nodes > 0, "--nodes must be nonzero");
        assert!(
            config.layout_group.is_none_or(|size| size > 0),
            "--layout-group must be nonzero"
        );
        assert!(
            config
                .frames
                .is_none_or(|frames| frames > config.warmup_frames),
            "--frames must exceed --warmup"
        );
        assert!(
            !matches!(
                config.workload,
                Workload::OneBoundaryPlacement | Workload::AllBoundaryPlacement
            ) || config.layout_group.is_some(),
            "boundary placement workloads require --layout-group"
        );
        assert!(
            !matches!(
                config.workload,
                Workload::OneBoundaryPlacement | Workload::AllBoundaryPlacement
            ) || config.geometry == Geometry::Grid,
            "boundary placement workloads currently use grid geometry"
        );
        config
    }
}

#[derive(Resource, Default)]
struct FrameTiming {
    frames: u32,
    milliseconds: Vec<f64>,
    main_start: Option<std::time::Instant>,
    main_milliseconds: Vec<f64>,
}

#[derive(Resource)]
struct StressNodes {
    nodes: Vec<StressItem>,
    boundaries: Vec<Entity>,
    image: Handle<Image>,
    alternate: bool,
}

#[derive(Clone, Copy)]
struct StressItem {
    entity: Entity,
    parent: Entity,
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
        .insert_resource(WinitSettings::continuous())
        .insert_resource(config.clone())
        .init_resource::<FrameTiming>()
        .add_systems(Startup, setup)
        .add_systems(First, start_main_time)
        .add_systems(Last, finish_main_time);

    if config.renderer == Renderer::Retained {
        app.add_plugins((UiRenderInfrastructurePlugin, RetainedUiRenderPlugin));
        app.add_systems(Update, report_main_work);
        app.sub_app_mut(RenderApp)
            .insert_resource(config.clone())
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
        Workload::OneBoundaryPlacement => match config.renderer {
            Renderer::Stock => {
                app.add_systems(Update, animate_one_stock_boundary);
            }
            Renderer::Retained => {
                app.add_systems(Update, animate_one_retained_boundary);
            }
        },
        Workload::AllBoundaryPlacement => match config.renderer {
            Renderer::Stock => {
                app.add_systems(Update, animate_all_stock_boundaries);
            }
            Renderer::Retained => {
                app.add_systems(Update, animate_all_retained_boundaries);
            }
        },
        Workload::OneChurn => {
            app.add_systems(Update, churn_one);
        }
    };
    if config.frames.is_some() {
        app.add_systems(Update, (record_frame_time, exit_after_frames).chain());
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
    let boundary_workload = matches!(
        config.workload,
        Workload::OneBoundaryPlacement | Workload::AllBoundaryPlacement
    );
    let group_size = config.layout_group.unwrap_or(config.nodes);
    let group_columns = (group_size as f32).sqrt().ceil() as usize;
    let group_count = config.nodes.div_ceil(group_size);
    let boundary_columns = (group_count as f32).sqrt().ceil() as usize;
    let mut nodes = Vec::with_capacity(config.nodes);
    let mut boundaries = Vec::new();
    let mut parent = root;
    for index in 0..config.nodes {
        if config
            .layout_group
            .is_some_and(|group_size| index.is_multiple_of(group_size))
        {
            let group = index / group_size;
            let members = group_size.min(config.nodes - index);
            let rows = members.div_ceil(group_columns);
            let mut boundary = commands.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: if boundary_workload {
                        px((group % boundary_columns * group_columns) as f32 * 7.0)
                    } else {
                        px(0)
                    },
                    top: if boundary_workload {
                        px((group / boundary_columns * group_columns) as f32 * 7.0)
                    } else {
                        px(0)
                    },
                    width: if boundary_workload {
                        px(group_columns as f32 * 7.0)
                    } else {
                        percent(100)
                    },
                    height: if boundary_workload {
                        px(rows as f32 * 7.0)
                    } else {
                        percent(100)
                    },
                    ..Default::default()
                },
                LayoutContainment,
                ChildOf(root),
            ));
            if config.renderer == Renderer::Retained
                && matches!(
                    config.workload,
                    Workload::OneBoundaryPlacement | Workload::AllBoundaryPlacement
                )
            {
                boundary.insert(RepaintBoundary::default());
            }
            parent = boundary.id();
            boundaries.push(parent);
        }
        let family = config.family.item(index);
        let item_index = if boundary_workload {
            index % group_size
        } else {
            index
        };
        let entity = spawn_item(
            &mut commands,
            parent,
            family,
            config.geometry,
            item_index,
            if boundary_workload {
                group_columns
            } else {
                columns
            },
            &image,
        );
        nodes.push(StressItem {
            entity,
            parent,
            family,
        });
    }
    commands.insert_resource(StressNodes {
        nodes,
        boundaries,
        image,
        alternate: false,
    });
}

fn spawn_item(
    commands: &mut Commands,
    parent: Entity,
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
        ChildOf(parent),
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
        Color::srgba(0.36, 0.42, 0.52, 0.5)
    } else {
        Color::srgba(0.34, 0.42, 0.52, 0.5)
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

fn animate_one_stock_boundary(
    mut stress: ResMut<StressNodes>,
    mut transforms: Query<&mut UiTransform>,
) {
    stress.alternate = !stress.alternate;
    transforms
        .get_mut(stress.boundaries[0])
        .unwrap()
        .translation
        .x = px(if stress.alternate { 1.0 } else { 0.0 });
}

fn animate_all_stock_boundaries(
    mut stress: ResMut<StressNodes>,
    mut transforms: Query<&mut UiTransform>,
) {
    stress.alternate = !stress.alternate;
    let x = px(if stress.alternate { 1.0 } else { 0.0 });
    for &boundary in &stress.boundaries {
        transforms.get_mut(boundary).unwrap().translation.x = x;
    }
}

fn animate_one_retained_boundary(
    mut stress: ResMut<StressNodes>,
    mut boundaries: Query<&mut RepaintBoundary>,
) {
    stress.alternate = !stress.alternate;
    boundaries
        .get_mut(stress.boundaries[0])
        .unwrap()
        .transform
        .translation
        .x = px(if stress.alternate { 1.0 } else { 0.0 });
}

fn animate_all_retained_boundaries(
    mut stress: ResMut<StressNodes>,
    mut boundaries: Query<&mut RepaintBoundary>,
) {
    stress.alternate = !stress.alternate;
    let x = px(if stress.alternate { 1.0 } else { 0.0 });
    for &boundary in &stress.boundaries {
        boundaries
            .get_mut(boundary)
            .unwrap()
            .transform
            .translation
            .x = x;
    }
}

fn churn_one(mut commands: Commands, config: Res<Config>, mut stress: ResMut<StressNodes>) {
    let old = stress.nodes[0];
    commands.entity(old.entity).despawn();
    let entity = spawn_item(
        &mut commands,
        old.parent,
        old.family,
        config.geometry,
        0,
        (config.nodes as f32).sqrt().ceil() as usize,
        &stress.image,
    );
    stress.nodes[0] = StressItem {
        entity,
        parent: old.parent,
        family: old.family,
    };
}

fn report_main_work(
    config: Res<Config>,
    main: Res<RetainedUiMainWorldCounters>,
    mut frames: Local<u32>,
) {
    *frames += 1;
    if config.frames.is_none() && (*frames).is_multiple_of(120) {
        info!("retained main={:?}", main.snapshot());
    }
}

fn report_render_work(
    paint: Res<RetainedUiPaintCounters>,
    layer: Res<RetainedUiLayerCounters>,
    config: Res<Config>,
    mut frames: Local<u32>,
    mut final_reported: Local<bool>,
) {
    *frames += 1;
    let report = config.frames.map_or_else(
        || (*frames).is_multiple_of(120),
        |limit| *frames >= limit && !*final_reported,
    );
    if report {
        *final_reported = true;
        info!(
            "retained paint={:?} layer={:?}",
            paint.snapshot(),
            layer.snapshot(),
        );
    }
}

fn record_frame_time(config: Res<Config>, time: Res<Time<Real>>, mut timing: ResMut<FrameTiming>) {
    timing.frames += 1;
    if timing.frames > config.warmup_frames {
        let milliseconds = time.delta_secs_f64() * 1_000.0;
        if milliseconds > 0.0 {
            timing.milliseconds.push(milliseconds);
        }
    }
}

fn start_main_time(mut timing: ResMut<FrameTiming>) {
    timing.main_start = Some(std::time::Instant::now());
}

fn finish_main_time(config: Res<Config>, mut timing: ResMut<FrameTiming>) {
    let elapsed = timing
        .main_start
        .take()
        .expect("First must run before Last")
        .elapsed();
    if timing.frames >= config.warmup_frames {
        timing
            .main_milliseconds
            .push(elapsed.as_secs_f64() * 1_000.0);
    }
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let rank = (percentile * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn report_frame_times(timing: &FrameTiming) {
    if timing.milliseconds.is_empty() {
        println!("frame-time: no samples");
        return;
    }

    let mut sorted = timing.milliseconds.clone();
    sorted.sort_by(f64::total_cmp);
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let over_4ms = timing
        .milliseconds
        .iter()
        .filter(|time| **time >= 4.0)
        .count();
    let mut longest_over_4ms = 0;
    let mut current_over_4ms = 0;
    for time in &timing.milliseconds {
        if *time >= 4.0 {
            current_over_4ms += 1;
            longest_over_4ms = longest_over_4ms.max(current_over_4ms);
        } else {
            current_over_4ms = 0;
        }
    }

    println!(
        "frame-time: samples={} mean={mean:.3}ms p50={:.3}ms p95={:.3}ms p99={:.3}ms \
         max={:.3}ms over_4ms={over_4ms} longest_over_4ms={longest_over_4ms}",
        sorted.len(),
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.95),
        percentile(&sorted, 0.99),
        sorted[sorted.len() - 1],
    );
    let main_mean =
        timing.main_milliseconds.iter().sum::<f64>() / timing.main_milliseconds.len() as f64;
    println!(
        "main-time: samples={} mean={main_mean:.3}ms",
        timing.main_milliseconds.len()
    );
}

fn exit_after_frames(
    config: Res<Config>,
    timing: Res<FrameTiming>,
    main: Option<Res<RetainedUiMainWorldCounters>>,
    mut frames: Local<u32>,
    mut exit: MessageWriter<AppExit>,
) {
    *frames += 1;
    if config.frames.is_some_and(|limit| *frames >= limit) {
        report_frame_times(&timing);
        if let Some(main) = main {
            println!("retained main={:?}", main.snapshot());
        }
        exit.write(AppExit::Success);
    }
}
