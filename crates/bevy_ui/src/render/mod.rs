mod pipeline;
mod render_pass;
mod ui_material;

use bevy_core_pipeline::{core_2d::Camera2d, core_3d::Camera3d};
use bevy_render::ExtractSchedule;

pub use pipeline::*;
pub use render_pass::*;
pub use ui_material::*;

use crate::{prelude::UiCameraConfig};
use bevy_app::prelude::*;
use bevy_asset::{load_internal_asset, HandleUntyped, Handle};
use bevy_ecs::prelude::*;
use bevy_math::{Mat4,UVec4, Vec2, Vec3, Vec4Swizzles};
use bevy_reflect::TypeUuid;
use bevy_render::{
    camera::Camera,
    render_graph::{RenderGraph, RunGraphOnViewNode, SlotInfo, SlotType},
    render_phase::{sort_phase_system, DrawFunctions, RenderPhase},
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
    view::{ExtractedView},
    Extract, RenderApp, RenderSet,
};

use bevy_transform::components::GlobalTransform;
use bytemuck::{Pod, Zeroable};
use std::hash::Hash;

pub mod node {
    pub const UI_PASS_DRIVER: &str = "ui_pass_driver";
}

pub mod draw_ui_graph {
    pub const NAME: &str = "draw_ui";
    pub mod input {
        pub const VIEW_ENTITY: &str = "view_entity";
    }
    pub mod node {
        pub const UI_PASS: &str = "ui_pass";
    }
}

pub const UI_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 13012847047162779583);

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum RenderUiSystem {
    ExtractNode,
}

pub struct UiRenderPlugin;

impl Plugin for UiRenderPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, UI_SHADER_HANDLE, "ui.wgsl", Shader::from_wgsl);

        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };

        render_app
            .init_resource::<UiImageBindGroups>()
            .init_resource::<UiMeta>()
            .init_resource::<ExtractedUiNodes>()
            .init_resource::<DrawFunctions<TransparentUi>>()
            .add_systems(
                (
                    extract_default_ui_camera_view::<Camera2d>,
                    extract_default_ui_camera_view::<Camera3d>,
                    reset_extracted_ui_nodes,

                )
                    .in_schedule(ExtractSchedule),
            )
            .add_system(prepare_uinodes.in_set(RenderSet::Prepare))
            .add_system(sort_phase_system::<TransparentUi>.in_set(RenderSet::PhaseSort));

        let ui_graph_2d = get_ui_graph(render_app);
        let ui_graph_3d = get_ui_graph(render_app);
        let mut graph = render_app.world.resource_mut::<RenderGraph>();

        if let Some(graph_2d) = graph.get_sub_graph_mut(bevy_core_pipeline::core_2d::graph::NAME) {
            graph_2d.add_sub_graph(draw_ui_graph::NAME, ui_graph_2d);
            graph_2d.add_node(
                draw_ui_graph::node::UI_PASS,
                RunGraphOnViewNode::new(draw_ui_graph::NAME),
            );
            graph_2d.add_node_edge(
                bevy_core_pipeline::core_2d::graph::node::MAIN_PASS,
                draw_ui_graph::node::UI_PASS,
            );
            graph_2d.add_slot_edge(
                graph_2d.input_node().id,
                bevy_core_pipeline::core_2d::graph::input::VIEW_ENTITY,
                draw_ui_graph::node::UI_PASS,
                RunGraphOnViewNode::IN_VIEW,
            );
            graph_2d.add_node_edge(
                bevy_core_pipeline::core_2d::graph::node::END_MAIN_PASS_POST_PROCESSING,
                draw_ui_graph::node::UI_PASS,
            );
            graph_2d.add_node_edge(
                draw_ui_graph::node::UI_PASS,
                bevy_core_pipeline::core_2d::graph::node::UPSCALING,
            );
        }

        if let Some(graph_3d) = graph.get_sub_graph_mut(bevy_core_pipeline::core_3d::graph::NAME) {
            graph_3d.add_sub_graph(draw_ui_graph::NAME, ui_graph_3d);
            graph_3d.add_node(
                draw_ui_graph::node::UI_PASS,
                RunGraphOnViewNode::new(draw_ui_graph::NAME),
            );
            // ui pass AFTER main pass
            graph_3d.add_node_edge(
                bevy_core_pipeline::core_3d::graph::node::MAIN_PASS,
                draw_ui_graph::node::UI_PASS,
            );
            // ui pass AFTER post processing
            graph_3d.add_node_edge(
                bevy_core_pipeline::core_3d::graph::node::END_MAIN_PASS_POST_PROCESSING,
                draw_ui_graph::node::UI_PASS,
            );
            // ui pass BEFORE upscaling
            graph_3d.add_node_edge(
                draw_ui_graph::node::UI_PASS,
                bevy_core_pipeline::core_3d::graph::node::UPSCALING,
            );
            // ui pass ... after... the 3d graph's input node? what
            // 2d does this too
            graph_3d.add_slot_edge(
                graph_3d.input_node().id,
                bevy_core_pipeline::core_3d::graph::input::VIEW_ENTITY,
                draw_ui_graph::node::UI_PASS,
                RunGraphOnViewNode::IN_VIEW,
            );
        }
    }
}

// composes the ui graph, which ultimately gets inserted as a subgraph?
fn get_ui_graph(render_app: &mut App) -> RenderGraph {
    // look into UiPassNode - it's got some kind of query in it
    let ui_pass_node = UiPassNode::new(&mut render_app.world);

    // empty render graph. the graph is just a map of nodes and subgraphs. there is a single input for whole graph
    let mut ui_graph = RenderGraph::default();

    // adding the node we made to the graph suggests that UiPassNode is actually a Render node? look into impl
    ui_graph.add_node(draw_ui_graph::node::UI_PASS, ui_pass_node);

    // inserting the ui graph's inputs returns the input node
    // figure out what's actually getting inserted? what is view entity, what are slots, etc
    let input_node_id = ui_graph.set_input(vec![SlotInfo::new(
        draw_ui_graph::input::VIEW_ENTITY,
        SlotType::Entity,
    )]);

    // this reads like it's taking UiPassNode::IN_VIEW and outptting that into VIEW_ENTITY
    // wtf?
    ui_graph.add_slot_edge(
        input_node_id,
        draw_ui_graph::input::VIEW_ENTITY,
        draw_ui_graph::node::UI_PASS,
        UiPassNode::IN_VIEW,
    );

    ui_graph
}

fn reset_extracted_ui_nodes(
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
) {
    extracted_uinodes.uinodes.clear();
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct UiVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

#[derive(Resource)]
pub struct UiMeta {
    vertices: BufferVec<UiVertex>,
    view_bind_group: Option<BindGroup>,
}

impl Default for UiMeta {
    fn default() -> Self {
        Self {
            vertices: BufferVec::new(BufferUsages::VERTEX),
            view_bind_group: None,
        }
    }
}

const QUAD_VERTEX_POSITIONS: [Vec3; 4] = [
    Vec3::new(-0.5, -0.5, 0.0),
    Vec3::new(0.5, -0.5, 0.0),
    Vec3::new(0.5, 0.5, 0.0),
    Vec3::new(-0.5, 0.5, 0.0),
];

const QUAD_INDICES: [usize; 6] = [0, 2, 3, 0, 1, 2];

pub fn prepare_uinodes(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut ui_meta: ResMut<UiMeta>,
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
) {
    ui_meta.vertices.clear();

    // sort by ui stack index, starting from the deepest node
    extracted_uinodes
        .uinodes
        .sort_by_key(|node| node.stack_index);

    let first = if let Some(first_node) = extracted_uinodes.uinodes.first() {
        first_node
    } else {
        // no nodes in the stack
        println!("no nodes in your ui stack?");
        return;
    };

    let mut start = 0;
    let mut end = 0;
    let mut current_batch_handle = first.image.clone_weak();
    let mut current_material_handle = first.material.clone_weak();
    let mut last_z = 0.0;
    for extracted_uinode in &extracted_uinodes.uinodes {
        if current_batch_handle != extracted_uinode.image || current_material_handle != extracted_uinode.material {
            if start != end {
                commands.spawn(UiBatch {
                    range: start..end,
                    image: current_batch_handle,
                    material: current_material_handle,
                    z: last_z,
                });
                start = end;
            }
            current_batch_handle = extracted_uinode.image.clone_weak();
            current_material_handle = extracted_uinode.material.clone_weak();
        }

        let uinode_rect = extracted_uinode.rect;
        let rect_size = uinode_rect.size().extend(1.0);

        // Specify the corners of the node
        let positions = QUAD_VERTEX_POSITIONS
            .map(|pos| (extracted_uinode.transform * (pos * rect_size).extend(1.)).xyz());

        // Calculate the effect of clipping
        // Note: this won't work with rotation/scaling, but that's much more complex (may need more that 2 quads)
        let positions_diff = if let Some(clip) = extracted_uinode.clip {
            [
                Vec2::new(
                    f32::max(clip.min.x - positions[0].x, 0.),
                    f32::max(clip.min.y - positions[0].y, 0.),
                ),
                Vec2::new(
                    f32::min(clip.max.x - positions[1].x, 0.),
                    f32::max(clip.min.y - positions[1].y, 0.),
                ),
                Vec2::new(
                    f32::min(clip.max.x - positions[2].x, 0.),
                    f32::min(clip.max.y - positions[2].y, 0.),
                ),
                Vec2::new(
                    f32::max(clip.min.x - positions[3].x, 0.),
                    f32::min(clip.max.y - positions[3].y, 0.),
                ),
            ]
        } else {
            [Vec2::ZERO; 4]
        };

        let positions_clipped = [
            positions[0] + positions_diff[0].extend(0.),
            positions[1] + positions_diff[1].extend(0.),
            positions[2] + positions_diff[2].extend(0.),
            positions[3] + positions_diff[3].extend(0.),
        ];

        let transformed_rect_size = extracted_uinode.transform.transform_vector3(rect_size);

        // Don't try to cull nodes that have a rotation
        // In a rotation around the Z-axis, this value is 0.0 for an angle of 0.0 or π
        // In those two cases, the culling check can proceed normally as corners will be on
        // horizontal / vertical lines
        // For all other angles, bypass the culling check
        // This does not properly handles all rotations on all axis
        if extracted_uinode.transform.x_axis[1] == 0.0 {
            // Cull nodes that are completely clipped
            if positions_diff[0].x - positions_diff[1].x >= transformed_rect_size.x
                || positions_diff[1].y - positions_diff[2].y >= transformed_rect_size.y
            {
                continue;
            }
        }

        let atlas_extent = extracted_uinode.atlas_size.unwrap_or(uinode_rect.max);
        let mut uvs = [
            Vec2::new(
                uinode_rect.min.x + positions_diff[0].x,
                uinode_rect.min.y + positions_diff[0].y,
            ),
            Vec2::new(
                uinode_rect.max.x + positions_diff[1].x,
                uinode_rect.min.y + positions_diff[1].y,
            ),
            Vec2::new(
                uinode_rect.max.x + positions_diff[2].x,
                uinode_rect.max.y + positions_diff[2].y,
            ),
            Vec2::new(
                uinode_rect.min.x + positions_diff[3].x,
                uinode_rect.max.y + positions_diff[3].y,
            ),
        ]
        .map(|pos| pos / atlas_extent);

        if extracted_uinode.flip_x {
            uvs = [uvs[1], uvs[0], uvs[3], uvs[2]];
        }
        if extracted_uinode.flip_y {
            uvs = [uvs[3], uvs[2], uvs[1], uvs[0]];
        }

        let color = extracted_uinode.color.as_linear_rgba_f32();
        for i in QUAD_INDICES {
            ui_meta.vertices.push(UiVertex {
                position: positions_clipped[i].into(),
                uv: uvs[i].into(),
                color,
            });
        }

        last_z = extracted_uinode.transform.w_axis[2];
        end += QUAD_INDICES.len() as u32;
    }

    // if start != end, there is one last batch to process
    if start != end {
        commands.spawn(UiBatch {
            range: start..end,
            image: current_batch_handle,
            material: current_material_handle,
            z: last_z,
        });
    }

    ui_meta.vertices.write_buffer(&render_device, &render_queue);
}





/// The UI camera is "moved back" by this many units (plus the [`UI_CAMERA_TRANSFORM_OFFSET`]) and also has a view
/// distance of this many units. This ensures that with a left-handed projection,
/// as ui elements are "stacked on top of each other", they are within the camera's view
/// and have room to grow.
// TODO: Consider computing this value at runtime based on the maximum z-value.
const UI_CAMERA_FAR: f32 = 1000.0;

// This value is subtracted from the far distance for the camera's z-position to ensure nodes at z == 0.0 are rendered
// TODO: Evaluate if we still need this.
const UI_CAMERA_TRANSFORM_OFFSET: f32 = -0.1;

#[derive(Component)]
pub struct DefaultCameraView(pub Entity);

pub fn extract_default_ui_camera_view<T: Component>(
    mut commands: Commands,
    query: Extract<Query<(Entity, &Camera, Option<&UiCameraConfig>), With<T>>>,
) {
    for (entity, camera, camera_ui) in &query {
        // ignore cameras with disabled ui
        if matches!(camera_ui, Some(&UiCameraConfig { show_ui: false, .. })) {
            continue;
        }
        if let (Some(logical_size), Some((physical_origin, _)), Some(physical_size)) = (
            camera.logical_viewport_size(),
            camera.physical_viewport_rect(),
            camera.physical_viewport_size(),
        ) {
            // use a projection matrix with the origin in the top left instead of the bottom left that comes with OrthographicProjection
            let projection_matrix =
                Mat4::orthographic_rh(0.0, logical_size.x, logical_size.y, 0.0, 0.0, UI_CAMERA_FAR);
            let default_camera_view = commands
                .spawn(ExtractedView {
                    projection: projection_matrix,
                    transform: GlobalTransform::from_xyz(
                        0.0,
                        0.0,
                        UI_CAMERA_FAR + UI_CAMERA_TRANSFORM_OFFSET,
                    ),
                    view_projection: None,
                    hdr: camera.hdr,
                    viewport: UVec4::new(
                        physical_origin.x,
                        physical_origin.y,
                        physical_size.x,
                        physical_size.y,
                    ),
                    color_grading: Default::default(),
                })
                .id();
            commands.get_or_spawn(entity).insert((
                DefaultCameraView(default_camera_view),
                RenderPhase::<TransparentUi>::default(),
            ));
        }
    }
}
