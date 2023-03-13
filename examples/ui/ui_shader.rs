//! This example illustrates how to create a button that changes color and text based on its
//! interaction state.

use bevy::{prelude::*, winit::WinitSettings, ui::{UiMaterial, UiMaterialPlugin, UiPipelineKey}, render::render_resource::{ShaderRef, RenderPipelineDescriptor, AsBindGroup}, reflect::TypeUuid};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(UiMaterialPlugin::<GradientUiMaterial>::default())
        .add_plugin(UiMaterialPlugin::<RoundUiMaterial>::default())
        // Only run the app when there is user input. This will significantly reduce CPU/GPU use.
        .insert_resource(WinitSettings::desktop_app())
        .add_startup_system(setup)
        .add_system(button_system)
        .run();
}

const NORMAL_BUTTON: Color = Color::rgb(0.15, 0.15, 0.15);
const HOVERED_BUTTON: Color = Color::rgb(0.25, 0.25, 0.25);
const PRESSED_BUTTON: Color = Color::rgb(0.35, 0.75, 0.35);

fn button_system(
    mut interaction_query: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (Changed<Interaction>, With<Button>),
    >,
    mut text_query: Query<&mut Text>,
) {
    for (interaction, mut color, children) in &mut interaction_query {
        //let mut text = text_query.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Clicked => {
                //text.sections[0].value = "Press".to_string();
                *color = PRESSED_BUTTON.into();
            }
            Interaction::Hovered => {
                //text.sections[0].value = "Hover".to_string();
                *color = HOVERED_BUTTON.into();
            }
            Interaction::None => {
                //text.sections[0].value = "Button".to_string();
                *color = NORMAL_BUTTON.into();
            }
        }
    }
}

#[derive(AsBindGroup, Clone, Copy, Default, TypeUuid, Resource)]
#[uuid = "15789298-5723-8944-8325-329755215813"]
pub struct GradientUiMaterial {
    #[uniform(0)]
    pub color_one: Color,
    #[uniform(0)]
    pub color_two: Color
}

impl UiMaterial for GradientUiMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path("shaders/gradient_ui_material.wgsl".into())
    }
}

#[derive(AsBindGroup, Clone, Copy, Default, TypeUuid, Resource)]
#[uuid = "15789298-5723-8944-8325-329755215814"]
pub struct RoundUiMaterial {
    #[uniform(0)]
    pub color: Color,
}

impl UiMaterial for RoundUiMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path("shaders/round_ui_material.wgsl".into())
    }
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>, mut gradient_materials: ResMut<Assets<GradientUiMaterial>>, mut circle_materials: ResMut<Assets<RoundUiMaterial>>) {
    // ui camera
    let gradient_ui_material = gradient_materials.add(GradientUiMaterial {
        color_one: Color::BLUE,
        color_two: Color::RED
    });

    let circle_ui_material = circle_materials.add(RoundUiMaterial{color: Color::PURPLE});

    commands.spawn(Camera2dBundle::default());
    commands
        .spawn(NodeBundle {
            style: Style {
                size: Size::width(Val::Percent(100.0)),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            ..default()
        })
        .insert(gradient_ui_material.clone())
        .with_children(|parent| {
            parent
                .spawn(NodeBundle {
                    style: Style {
                        size: Size::new(Val::Px(150.0), Val::Px(65.0)),
                        ..default()
                    },
                    background_color: NORMAL_BUTTON.into(),
                    ..default()
                })
                .insert(gradient_ui_material.clone());

            parent
                .spawn(NodeBundle {
                    style: Style {
                        size: Size::new(Val::Px(150.0), Val::Px(65.0)),
                        ..default()
                    },
                    background_color: BackgroundColor(Color::LIME_GREEN),
                    ..default()
                });

            parent
                .spawn(NodeBundle {
                    style: Style {
                        size: Size::new(Val::Px(150.0), Val::Px(65.0)),
                        ..default()
                    },
                    background_color: BackgroundColor(Color::LIME_GREEN),
                    ..default()
                })
                .insert(circle_ui_material.clone());



        });
}
