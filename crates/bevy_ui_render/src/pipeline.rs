use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_ecs::prelude::*;
use bevy_mesh::VertexBufferLayout;
use bevy_render::{
    render_resource::{
        binding_types::{sampler, storage_buffer_read_only_sized, texture_2d, uniform_buffer},
        *,
    },
    view::ViewUniform,
};
use bevy_shader::Shader;
use bevy_utils::default;

#[derive(Resource)]
pub struct UiPipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub image_layout: BindGroupLayoutDescriptor,
    pub instance_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

pub fn init_ui_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "ui_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );

    let image_layout = BindGroupLayoutDescriptor::new(
        "ui_image_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );
    let instance_layout = BindGroupLayoutDescriptor::new(
        "ui_instance_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    );

    commands.insert_resource(UiPipeline {
        view_layout,
        image_layout,
        instance_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "ui.wgsl"),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiPipelineKey {
    pub target_format: TextureFormat,
    pub anti_alias: bool,
    pub storage_buffers: bool,
}

impl SpecializedRenderPipeline for UiPipeline {
    type Key = UiPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();
        if key.anti_alias {
            shader_defs.push("ANTI_ALIAS".into());
        }
        if key.storage_buffers {
            shader_defs.push("UI_STORAGE_INSTANCE".into());
        }

        let buffers = if key.storage_buffers {
            vec![VertexBufferLayout::from_vertex_formats(
                VertexStepMode::Instance,
                vec![VertexFormat::Uint32],
            )]
        } else {
            vec![ui_geometry_vertex_layout(), ui_style_vertex_layout()]
        };
        let mut layout = vec![self.view_layout.clone(), self.image_layout.clone()];
        if key.storage_buffers {
            layout.push(self.instance_layout.clone());
        }

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers,
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            layout,
            label: Some("ui_pipeline".into()),
            ..default()
        }
    }
}

pub(crate) fn ui_geometry_vertex_layout() -> VertexBufferLayout {
    VertexBufferLayout::from_vertex_formats(
        VertexStepMode::Instance,
        vec![
            // transform columns and translation
            VertexFormat::Float32x2,
            VertexFormat::Float32x2,
            VertexFormat::Float32x2,
            // size
            VertexFormat::Float32x2,
            // world-space clipping offsets
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            // texture coordinates
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
        ],
    )
}

pub(crate) fn ui_style_vertex_layout() -> VertexBufferLayout {
    VertexBufferLayout {
        array_stride: 112,
        step_mode: VertexStepMode::Instance,
        attributes: vec![
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 0,
                shader_location: 8,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 16,
                shader_location: 9,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 32,
                shader_location: 10,
            },
            VertexAttribute {
                format: VertexFormat::Uint32x4,
                offset: 48,
                shader_location: 11,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 64,
                shader_location: 12,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 80,
                shader_location: 13,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 96,
                shader_location: 14,
            },
        ],
    }
}
