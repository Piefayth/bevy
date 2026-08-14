# Retained UI design record

This document records the research and design constraints for a damage-tracked
retained renderer for Bevy UI. It records both the governing contract and the
capabilities already established by executable proofs.

The objective is stricter than ordinary "retained mode":

- an unchanged UI entity is not extracted again;
- unchanged paint commands and GPU data are retained;
- pixels are repainted only when a complete dependency proof says their current
  value may be wrong;
- layout, placement, paint, raster, and composition changes are independent;
- a static UI performs no Taffy computation, recursive geometry, stack, clipping,
  extraction, or raster work; stock `Changed<T>` candidate scans and, if the game
  renders a fresh world frame, the irreducible composition of visible cached UI
  remain;
- correctness never depends on a timing threshold, heartbeat, or cache
  promotion heuristic.

## Scope decision

Bevy 0.19 deliberately separates `bevy_ui` from `bevy_ui_render`. The
`ui_api`/`ui_bevy_render` features and the separate `UiPlugin` and
`UiRenderPlugin` make an alternative renderer an intended use case. See
[PR #18703](https://github.com/bevyengine/bevy/pull/18703).

The first implementation is therefore an external replacement for
`bevy_ui_render`. Crate scope expands only when a tested requirement cannot be
implemented through public scheduling and render-graph APIs.

The implemented expansion decisions are:

1. `bevy_ui`: stock layout, stack construction, and clipping walked static trees,
   and stock Taffy invalidation coupled every descendant to its complete UI root.
   [Issue #22909](https://github.com/bevyengine/bevy/issues/22909) describes the
   same idle cost and root invalidation. A focused `bevy_ui` patch was required
   for independent layout/geometry scheduling and semantic layout and paint
   containment.
2. `bevy_core_pipeline`: a standalone retained crate can add a separate
   composite pass, but cannot fold the retained layer into Bevy's existing
   final blit. The implementation adds an optional premultiplied overlay input
   to that blit. Views without an overlay still select the original two-binding
   pipeline. `bevy_post_process` only supplies the new false pipeline-key field
   for MSAA writeback.

Focused changes remain in the crates whose behavior they own.
`bevy_ui` now caches an image node's intrinsic-size inputs, so changing only an
image tint does not falsely rewrite `ContentSize` and wake layout.
`bevy_ui_render` exposes its ordinary-node preparation as a removable public
system set and gives the shared gradient shader a public import path. The
replacement can therefore install persistent node preparation and reuse Bevy's
gradient functions without keeping an unused transient upload. Repaint
boundaries also propagate a generic `ComputedUiPaintTarget`; the stock generic
`UiMaterial` extractor honors it, localizes transform and clip into that
target's coordinate space, and records immediate-mode target volatility.
That small cross-crate hook is necessary because an arbitrary material plugin is
monomorphized outside the replacement crate: without it, a material below a
boundary would queue onto the camera phase and bypass the cached surface.
Neither change requires `bevy_core_pipeline`.

Current upstream UI-render work retains intermediate render data, but not final
pixels:

- [PR #24893](https://github.com/bevyengine/bevy/pull/24893): retained extracted
  render entities;
- [PR #25290](https://github.com/bevyengine/bevy/pull/25290): retained phase
  items;
- [PR #25191](https://github.com/bevyengine/bevy/pull/25191): retained vertex
  allocation;
- [goal #25149](https://github.com/bevyengine/bevy/issues/25149): overall UI
  rendering improvement goal.

These are compatible foundations, but they do not implement raster retention or
damage repair.

### Scope probe results

An external plugin can attach a run condition to a `UiSystems` set after the
systems in that set were registered. The red/green regression test is
`bevy_ui_render_retained/tests/ui_systems_scope.rs`. This is enough to stop all
layout work for a wholly quiet tree without changing `bevy_ui`.

It was not enough to separate layout from placement externally.
`ui_layout_system` performed both Taffy layout and the recursive update of
`ComputedNode`/`UiGlobalTransform`, including transforms and scroll offsets.
That recursive update was not a separately schedulable public system, and the
`UiSurface` holding the Taffy state was private. Therefore a transform-only or
scroll-only animation could avoid Taffy only by either:

- moving that geometry update into an independently scheduled `bevy_ui`
  system; or
- duplicating Bevy's geometry traversal in the third-party crate.

The second choice creates two owners for the same derived state and is rejected.
The focused `bevy_ui` patch splits the original function into public,
ordered `ui_layout_system` and `ui_geometry_system` systems. The former owns
Taffy synchronization and computation; the latter owns placement, scrolling,
rounding, outlines, radii, and derived render geometry. This is the second
justified `bevy_ui` expansion: it lets a `UiTransform` or scroll animation skip
Taffy without duplicating Bevy internals. Geometry consumes the exact scopes that
layout actually computed instead of independently inferring them from another
`Changed<T>` list. Transforms and scrolling resolve only the changed subtree;
radius and outline changes resolve one node.

Root-level gating alone cannot remove a real layout spike. In a flat 10,000-child
flex row, changing one child's width can change every sibling through flex shrink;
Taffy is correct to recompute that dependency root. Profiling on the development
machine attributed roughly 3.5--4 ms of the original 4--5 ms frame to that
coupled Taffy computation. Disabling retained bookkeeping did not remove it.

`LayoutContainment` is the justified third `bevy_ui` expansion. It is an explicit
size-and-layout promise, analogous to CSS size/layout containment and Flutter's
relayout boundaries: descendants are laid out under a disconnected Taffy root
whose constraints are the boundary's already-resolved border box. Descendant
changes therefore cannot affect the boundary's size or anything outside it.
The boundary's own `Node` still participates normally in its parent's layout.
No threshold or inferred promotion is involved; applications should give a
boundary an explicit size when a zero intrinsic size is not useful.

```rust
commands
    .spawn((
        Node {
            width: px(640),
            height: px(360),
            ..default()
        },
        LayoutContainment,
    ))
    .with_children(spawn_menu_contents);
```

Dirty layout scopes distinguish the boundary's outer box from its contained
contents. This matters when a containment boundary is itself an ECS root:
changing a descendant computes only the contained tree, while changing the
boundary's own size computes the outer tree and then the contained tree if its
constraints changed. Nested containment stops at the nearest boundary, and
reparenting updates exactly the old and new boundaries. Equal `Node` writes are
compared against Taffy's canonical style and do not dirty Taffy.

`PaintContainment` is a separate semantic promise. Descendant paint and
`OverrideClip` may not escape the boundary's border box. `CalculatedClip`
therefore carries two exact clip chains: the final parent-facing clip and the
boundary-local raster clip. The local chain restarts at each nested paint
boundary, while the parent chain keeps accumulating normally. Keeping the two
channels is necessary: a cached subtree must rasterize pixels hidden by its
current parent clip so a later compositor transform can reveal them, but it
must never rasterize pixels outside its own paint promise.

`RepaintBoundary` requires `PaintContainment` but deliberately does not require
`LayoutContainment`. Raster caching must not silently change intrinsic sizing
or flex dependencies. Applications combine both components when they can make
both promises. As with other Bevy required components, removing
`RepaintBoundary` does not remove an existing `PaintContainment`; remove both
components explicitly when both semantics should end.

The retained source is exactly the boundary's border box. Its own paint is
therefore clipped there too; put an exterior shadow or outline on a parent
wrapper. This makes the surface size a declared, stable quantity instead of a
content-dependent allocation that can jump when an effect or overflowing child
changes. The behavior is part of the boundary contract, not inferred promotion
or a renderer threshold.

```rust
commands.spawn((
    Node {
        width: px(640),
        height: px(360),
        ..default()
    },
    LayoutContainment,
    RepaintBoundary::from_translation(Val2::px(0, 0)),
));
```

Animating `RepaintBoundary::transform` or `opacity` changes only composition.
Animating the node's ordinary `UiTransform` remains a layout-tree placement
change and updates descendant `UiGlobalTransform`s.

There is a second possible `bevy_ui` boundary for O(changes) candidate
nomination. In Bevy 0.19, lifecycle hooks run for component insertion,
replacement, and removal, but not for ordinary writes through `Mut<T>`. The
executable proof is `bevy_ui_render_retained/tests/mutation_scope.rs`.

Keeping stock components and `Changed<T>` is nevertheless an exact pull model,
not a correctness compromise. It nominates changed entities without
re-extracting unchanged entities, but its filters scan every matching entity.
On this development machine a quiet release-mode query over 10,000 entities
measured approximately 10--12 microseconds for one `Changed` input and 29--31
microseconds for an eight-input `Or`, including roughly 1.3 microseconds of
schedule overhead. At 1,000 entities the eight-input case measured about 5
microseconds. These are short local comparison points, not portable claims; the
source is `benches/benches/bevy_ui/changed.rs`.

The real constraint is dependency completeness. Canonical paint comparison
prevents a falsely nominated entity from causing damage, but cannot recover an
entity that a missing `Changed<T>` input never nominated. Each paint family
must therefore declare its complete component, asset, resource, removal, and
cross-entity dependencies next to the code that builds its canonical record,
with a mutation-matrix test for every declared input. A mutable dereference that
writes an equal value is a harmless false nomination: record comparison pays
the extraction for that entity but produces no upload or damage.

Accepting the measured O(matching entities x tested inputs) scan keeps the
renderer third-party and avoids changing Bevy's component mutation model. If
that scan is later rejected, the structurally correct alternatives are an ECS
mutation journal or immutable render-affecting values replaced through an
invalidating API. Manual dirty flags and periodic audits remain rejected
because they permit permanent stale pixels.

The main-world quiescence plugin replaces exactly four stock systems:
`ui_layout_system`, `ui_geometry_system`, `ui_stack_system`, and
`update_clipping_system`. Each keeps its original public set membership and
gains a complete changed/removed-input condition. Other application systems
placed in `UiSystems::Layout`, `UiSystems::Stack`, or `UiSystems::PostLayout`
remain ungated. Exact replacement requires naming the stock function;
`ui_stack_system` was private, so the smallest initial `bevy_ui` patch publicly
re-exported it and deleted its private cache wrapper in favor of the identical
`Local<Vec<Vec<_>>>`. Gating the whole public set from a third-party crate was
rejected because it silently changes unrelated systems' execution semantics.

The core-pipeline probe found that a third-party plugin can remove Bevy's public
`upscaling` system and install a correct final writer; the executable proof is
`bevy_ui_render_retained/tests/core_pipeline_scope.rs`. That is a workable
zero-vendoring fallback, but it duplicates Bevy's output semantics and owns a
large rebase surface. A separate composite pass is simpler but adds another
attachment load/store.

The selected implementation instead makes Bevy's existing final blit accept an
optional premultiplied overlay. The retained pass repairs its private surfaces
before `upscaling`, publishes a texture only for a visible committed layer, and
the unchanged final writer selects plain or fused pipeline and bind-group
layouts. `CameraOutputMode::Skip` returns before retained surface work and
before the final writer, while owed damage remains intact. This is the first
justified expansion beyond `bevy_ui_render`: it removes an otherwise
irreducible extra fullscreen pass and keeps one owner for output blending,
color conversion, attachment acquisition, and presentation. GPU differentials
prove the no-UI path, premultiplied UI, nonzero viewports, default multi-camera
alpha composition, and `CameraOutputMode::Skip` byte-identical to stock on an
image target. Actual window presentation still requires one integration test
on each supported backend.

Replacing individual stock extraction systems exposed one smaller
`bevy_ui_render` composition boundary. `Schedule::remove_systems_in_set` first
initializes a changed schedule, but `MainWorld` exists in the render world only
while `ExtractSchedule` is running. A third-party plugin therefore cannot
safely remove a stock extractor during plugin construction. Keeping the old
system behind a false run condition or temporarily moving the main world were
rejected.

The single-crate patch is `UiRenderInfrastructurePlugin`. It owns Bevy's UI
shader, camera extraction, phase, queue, batching, and buffer preparation but
no paint extraction or UI pass. Stock `UiRenderPlugin` composes that
infrastructure with its existing extractors and pass; the retained crate uses
the infrastructure directly. This preserves a maintainable third-party
renderer while requiring a small patch only to the crate it replaces. No
`bevy_ecs` patch is required. The optional standalone-composite version could
stop here; the fused implementation additionally carries the focused
`bevy_core_pipeline` change described above.

The infrastructure no longer serializes unrelated paint-family extractors.
They all wait for camera extraction, then run concurrently, followed by one
explicit retained apply/propagate barrier. Bevy's stock text background,
shadow, glyph, and cursor extractors remain chained in that order: the GPU
differential suite proved their shared transient representation is
order-sensitive. Ambiguity detection is an error in every GPU test, so the
parallel graph cannot silently acquire another unordered resource conflict.

Sliced and tiled images exposed the same boundary one level down: Bevy's
`UiTextureSlicerPlugin` coupled stock extraction to reusable pipeline setup and
preparation. It is now composed from a public
`UiTextureSlicerInfrastructurePlugin` plus the stock extractor and batched
queue. The retained renderer uses the infrastructure with its own exact-item
queue. `DrawUiTextureSliceItem` prepares and draws one retained quad, so damage
culling cannot silently submit adjacent same-texture slices as a stock batch.
This remains a focused `bevy_ui_render` patch; it does not expand crate scope.

Gradients use an equivalent `GradientInfrastructurePlugin` split. The shared
`resolve_gradient` function is the sole implementation of logical stop,
radial-shape, and conic-angle resolution for both stock and retained extraction.
A red-first mixed-stack test exposed a stock ordering defect: a one-stop
gradient was converted to the ordinary node pipeline and could no longer keep
its list position among multistop gradients queued by another system. The
optimization was removed. Resolution now represents a solid gradient as two
equal stops, so every entry uses one ordered gradient pipeline and the retained
renderer needs no special case.

Box shadows share the public `resolve_box_shadow` function. The retained crate
owns candidate extraction, canonical records, persistent instances, queueing,
and its damage-mask-aware shader. Stock and retained paths still have one
implementation of target-relative value resolution. This remains entirely
inside the focused `bevy_ui_render` replacement boundary.

`ViewportNode` needs no additional Bevy patch. It resolves to the existing UI
image command, so the retained crate replaces only its extractor and reuses the
same node pipeline and sampled-image dependency index.

Custom `UiMaterial` exposed one final reusable boundary inside
`bevy_ui_render`. `UiMaterialPlugin<M>` now composes a public
`UiMaterialInfrastructurePlugin<M>` with stock extraction, and material
preparation emits one type-erased vertex-range component beside its typed
batch. The retained crate can therefore reuse arbitrary material pipelines and
draw commands without a callback registry or knowledge of `M`. Existing
`UiMaterialPlugin<M>` integrations remain correct under the retained renderer:
unknown material phase items conservatively force a full-target repaint every
frame. Replacing that plugin with `RetainedUiMaterialPlugin<M>` opts into exact
retention. This is a focused `bevy_ui_render` composition patch, not a reason
to vendor `bevy_ui` or `bevy_core_pipeline`.

## Model learned from other UI systems

Mature UI systems retain several representations rather than one:

```text
application / ECS
    structure and state dependencies
layout tree
    measure and placement
paint artifact
    display commands and stable paint identity
raster surfaces
    physical coverage and pixel damage
compositor scene
    transforms, opacity, clips, scroll, and surfaces
presented buffers
    logical frame damage plus recycled-buffer history
```

Each transition has its own dirty domain. A single dirty bit cannot express the
minimum correct work.

### Flutter

Flutter separates layout, paint, compositing-bit, and composited-layer updates.
A repaint boundary gives a subtree a separate display list; an engine layer can
be retained across scenes; raster caching is an additional optimization rather
than the definition of retention.

- [`RenderObject.markNeedsPaint`](https://api.flutter.dev/flutter/rendering/RenderObject/markNeedsPaint.html)
- [`markNeedsCompositedLayerUpdate`](https://api.flutter.dev/flutter/rendering/RenderObject/markNeedsCompositedLayerUpdate.html)
- [`RenderRepaintBoundary`](https://api.flutter.dev/flutter/rendering/RenderRepaintBoundary-class.html)
- [`SceneBuilder.addRetained`](https://api.flutter.dev/flutter/dart-ui/SceneBuilder/addRetained.html)

Flutter also distinguishes this frame's damage from damage already present in a
recycled framebuffer. Its Metal backend accumulates damage separately for each
drawable texture. This is the exact solution to multi-buffer persistence; a
fixed "settling" duration is not.

- [Flutter rasterizer damage](https://github.com/flutter/flutter/blob/master/engine/src/flutter/shell/common/rasterizer.cc)
- [Metal drawable damage history](https://github.com/flutter/flutter/blob/master/engine/src/flutter/shell/gpu/gpu_surface_metal_skia.mm)

### Chromium

Chromium retains stable display items and paint chunks, then associates them
with four property trees: transform, clip, effect, and scroll. Raster
invalidation, compositor damage, and presentation damage are distinct. The
active/pending compositor trees preserve an old complete scene until a new one
is ready.

- [Blink paint artifacts and property trees](https://chromium.googlesource.com/chromium/src/%2B/HEAD/third_party/blink/renderer/platform/graphics/paint/README.md)
- [Compositor architecture](https://chromium.googlesource.com/chromium/src.git/%2B/master/docs/how_cc_works.md)
- [Compositor animations](https://chromium.googlesource.com/chromium/src/%2B/HEAD/third_party/blink/renderer/core/animation/README.md)

Appearing content damages its new bounds, disappearing content its old bounds,
and reordering/property changes generally damage old and new coverage. Transform,
opacity, scrolling, and some filters can be updated by the compositor without
rebuilding paint.

### Firefox WebRender

WebRender turns a display list into picture, spatial, and clip trees. Cached
picture tiles track all inputs that can affect them: primitives, clips, image
keys, opacity bindings, and transforms. Invalidated dependency leaves form an
exact tile dirty rectangle.

- [Firefox rendering overview](https://searchfox.org/firefox-main/source/gfx/docs/RenderingOverview.rst)
- [WebRender picture caching](https://searchfox.org/firefox-main/source/gfx/wr/webrender/src/picture.rs)

This validates explicit resource and property dependencies. WebRender also uses
heuristics to choose cache slices; those heuristics affect performance, not
correctness, and are not part of our initial design.

### Android Views and Jetpack Compose

Android `RenderNode` stores a display list separately from properties such as
translation, scale, rotation, and alpha. A content change rerecords the affected
node; a property animation does not. HWUI damages both the old and new node
state and joins historical damage according to buffer age.

- [Hardware-accelerated drawing model](https://developer.android.com/develop/ui/views/graphics/hardware-accel)
- [`RenderNode`](https://developer.android.com/reference/android/graphics/RenderNode)
- [HWUI buffer-age damage](https://android.googlesource.com/platform/frameworks/base/%2B/517a9583a869ed1913efeb003592db184d1a09e0/libs/hwui/renderthread/CanvasContext.cpp)

Compose tracks state reads by execution phase. Composition, measurement,
placement, and drawing have restart scopes. Reading an animated color during
drawing can restart drawing alone; reading an offset during placement can avoid
composition and measurement.

- [Compose phases](https://developer.android.com/develop/ui/compose/phases)
- [Compose phase performance](https://developer.android.com/develop/ui/compose/performance/phases)

This is the closest model for Bevy component-to-dirty-domain mapping.

### UIKit and Core Animation

Core Animation stores view content in layer backing bitmaps and maintains model,
presentation, and render layer trees. Cached content can be transformed and
faded without asking the view to redraw. Layout and display have separate dirty
APIs, and transactions commit layer-tree changes atomically.

- [Core Animation basics](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/CoreAnimation_guide/CoreAnimationBasics/CoreAnimationBasics.html)
- [`CALayer`](https://developer.apple.com/documentation/quartzcore/calayer)
- [`CATransaction`](https://developer.apple.com/documentation/quartzcore/catransaction)

### Unity, Unreal, Qt, and WPF

Unity UI Toolkit retains its visual tree, vertex allocations, and batches. Its
`DynamicTransform`, `GroupTransform`, and `DynamicColor` usage hints avoid
unnecessary geometry regeneration. This demonstrates that retained geometry and
retained final pixels are separate features in a continuously rendered game.

- [Unity `UsageHints`](https://docs.unity3d.com/6000.0/ScriptReference/UIElements.UsageHints.html)
- [UI Toolkit performance](https://docs.unity3d.com/Manual/best-practice-guides/ui-toolkit-for-advanced-unity-developers/optimizing-performance.html)

Unreal makes that distinction explicit: invalidation caches hierarchy, layout,
and paint data, while a Retainer Panel additionally flattens a subtree into a
texture. Retainer repaint is expensive and consumes extra memory; volatile
widgets deliberately bypass paint caching.

- [Slate and UMG invalidation](https://dev.epicgames.com/documentation/unreal-engine/invalidation-in-slate-and-umg-for-unreal-engine)

Qt Quick synchronizes only changed items into a render-thread scene graph and
retains GPU geometry and batch roots. WPF similarly stores serialized retained
drawing data.

- [Qt Quick scene graph](https://doc.qt.io/qt-6/qtquick-visualcanvas-scenegraph.html)
- [Qt Quick renderer](https://doc.qt.io/qt-6/qtquick-visualcanvas-scenegraph-renderer.html)
- [WPF retained rendering](https://learn.microsoft.com/en-us/dotnet/desktop/wpf/graphics-multimedia/wpf-graphics-rendering-overview)

CSS containment demonstrates that a boundary must be a semantic promise. Layout
or paint containment permits optimization precisely because effects are
guaranteed not to escape the boundary.

- [CSS Containment](https://www.w3.org/TR/css-contain-2/)

## Dirty domains

The implementation uses these conceptual domains:

| Domain | Examples | Required work |
| --- | --- | --- |
| Structure | children, visibility participation, render family | rebuild affected identities/order/dependencies |
| Measure | layout fields in `Node`, content size, text metrics, target scale | run layout for the nearest semantic dependency scope |
| Placement | `UiTransform`, scroll offset | update spatial properties and affected descendant coverage |
| Paint | colors, glyphs, borders, image selection | rebuild only affected paint records and repair their pixels |
| Composite | retained-island transform or opacity | update compositor properties, no island repaint |
| Resource | image/atlas/material/sampled-surface revision | repaint exact readers and their filter reach |

For existing Bevy components, the intended semantics are:

- layout fields in `Node` nominate layout, then canonical Taffy-style comparison
  decides whether layout actually became dirty;
- `LayoutContainment` prevents descendant measure changes from escaping its
  explicitly sized box;
- `UiTransform` changes placement but not measurement;
- `BackgroundColor`, `BorderColor`, and equivalent visual components change
  paint but not layout;
- `ZIndex`, `GlobalZIndex`, hierarchy, and render-family changes affect order;
- a declared retained boundary allows subtree transform and opacity to become
  composite-only changes.

## Retained render representation

Every drawable has a stable `PaintId`. Its retained paint record contains every
input the renderer consumes:

- family-specific command data;
- paint order;
- transform, clip, and effect property references;
- exact conservative physical-pixel coverage;
- GPU allocation/batch ownership;
- sampled resource IDs, exact content revisions, and read regions.

Change notification only queues possible work. When a queued entity is
extracted, its canonical new paint records are compared with the retained old
records. Equal records do nothing. Changed records damage old union new
coverage.

We do **not** scan or fingerprint every extracted item each frame. That would be
correct but would violate the extraction objective. A full reference extraction
is retained only as a test oracle.

Custom draw families must declare complete coverage and resource dependencies.
The safe fallback for an incomplete contract is volatility: repaint it whenever
its containing view renders. Under-declared dependencies are never accepted as
an optimization.

## Raster surfaces and repair

A window needs a UI-owned persistent surface because its presentation textures
rotate. An image render target persists when its entire camera is inactive, as
proved by
`stock_image_target_persists_only_when_the_whole_camera_is_inactive`. Skipping
only Bevy's UI pass is insufficient: the remaining Core2d/upscaling work
overwrites the target with transparent pixels. A retained offscreen camera must
therefore become inactive after a completed repair, or use a render graph that
does no target write on a quiet frame.

One atomic repair does the following:

1. Preflight every pipeline, bind group, texture, and buffer needed by the
   repair.
2. Keep old retained state and damage owed if preflight fails.
3. Rasterize the exact rectangle union once into an `R8Unorm` damage mask.
4. Wipe that union with blending disabled.
5. Replay every intersecting retained record once, bottom-up; retained core,
   gradient, and shadow shaders discard fragments outside the mask.
6. Replay arbitrary stock slice/material pipelines under exact region scissors,
   because third-party pipelines cannot be required to bind the mask.
7. Commit the new retained state only after commands for the complete repair
   were encoded.

All repairs encoded in one render frame append their mask/copy/wipe rectangles
to one immutable-range GPU arena. Reusing offset zero for each surface was
proven wrong by a sustained boundary-content animation: a later parent repair
overwrote the child repair's vertex data before the command encoder executed,
and presented an L-shaped mixture of two color generations. The arena is
cleared once before render-graph execution, never between surface repairs.

Stencil was tested and rejected. Adding a stencil attachment changes render
pipeline compatibility, so every stock and third-party material pipeline would
need a matching depth/stencil specialization. A sampled color mask preserves
exact pixels for the renderer-owned families without expanding that contract.
It also lets a moved item draw once across disjoint old and new regions instead
of once per scissor.

The region representation is a canonical union of integer physical rectangles.
No subpixel tolerance is required. A record's conservative coverage includes
antialiasing, shadows, outlines, filters, and texture sampling reach. Old
coverage remains authoritative until the new pixels commit, so subpixel drift
cannot accumulate invisibly.

Candidate selection uses a persistent exact two-dimensional bounds tree.
Geometry changes refit only affected leaves and ancestors. Direct dirty groups
are tracked by a fixed-size earliest/latest epoch interval, so dense changes can
skip spatial work while delayed resources cannot grow unbounded history. The
tree is brought current before the next partial spatial query; no stale bound is
ever queried.

Offscreen surfaces carry monotonically increasing content generations. Readers
declare the sampled surface and source region. A source commit invalidates the
mapped reader coverage exactly. This replaces fixed settling frames and
heartbeats.

Surface loss, resize, format/scale changes, atlas eviction, or device-resource
loss invalidate the affected retained surface completely.

## Composition

If the world below the UI changes, cached UI must be composited again. This is
irreducible unless the operating-system compositor owns the UI as a separate
surface, which portable `wgpu` does not expose.

The first proposed co-draw was:

```text
draw world/upscaled image
draw cached occupied UI regions
```

That is not generally equivalent to Bevy's output semantics. A later camera can
blend its complete world-plus-UI source into an existing target with an
arbitrary `CameraOutputMode` blend state. Drawing world and UI separately would
apply that output blend twice and can double-multiply alpha. Restricting the
optimization to the first or replace-mode camera was rejected as a parallel
correctness path.

The selected focused `bevy_core_pipeline` integration instead samples the
premultiplied retained layer in Bevy's existing fullscreen final blit, combines
it with the world, then applies color-space conversion and the camera output
blend once. This deletes the standalone UI composite pass and its attachment
load/store. The price is one additional UI texture read for every output pixel
while UI is visible, even outside occupied UI bounds. That trade is expected to
favor tile GPUs but is not called free and must be included in the
physical-device sweep. A camera with no visible UI selects the original plain
shader entry, performs the same single world-texture draw as stock, and
allocates no retained surface.

The public `wgpu::Surface` presentation API has no damage-region parameter, so
portable platform partial-present behavior cannot currently be promised.

- [`wgpu::Surface`](https://docs.rs/wgpu/latest/wgpu/struct.Surface.html)

A general subtree repaint boundary cannot be implemented by cutting a hole in
one cached parent texture and compositing the subtree afterward. A later sibling
may paint above the boundary; flattening all non-boundary paint into one texture
loses the ordering point at which the boundary layer must be inserted. A
topmost-only compositor is therefore not the general primitive and is rejected.

The implemented representation is an ordered retained compositor. A parent
containing declared boundaries stores each nonempty contiguous ordinary-paint
run between boundaries in a tightly bounded retained source, and alternates
those sources with boundary sources in exact paint order. It damage-composes
that list into one flat presentation surface. This gives the required
contracts:

- a parent with no boundary collapses to the existing single retained surface;
- a quiet parent executes no paint or compositor pass and presents one cached
  texture, independent of its historical boundary count;
- boundary placement changes rasterize no paint record and compose only the
  exact old/new output damage from cached sources;
- arbitrary sibling interleaving and nested atomic stacking contexts remain
  exact.

The boundary source has its own paint-local translation and stack origin.
Reflow that uniformly moves or renumbers the subtree therefore changes only the
parent composite record; surviving source pixels and their local order remain
unchanged. A three-toast removal/reflow proof moves both survivors, erases the
departed source, and rasterizes zero paint.

Ordinary run surfaces keep a stable physical coordinate space. If paint moves
past a run's old crop, the allocation grows to enclose the new coverage, copies
the shared old pixels into both replacement slots, and repairs only the paint's
exact old/new damage. It neither clips the move nor rerasterizes an unchanged
sibling. Run allocations never shrink while their ordered run survives, so the
cost is explicit and bounded by the parent target. `resize_copy_pixels` reports
the two-slot copy separately from paint.

The parent display list has one persistent exact spatial index. A changed
parent queries it for currently intersecting groups, unions directly changed
groups in an O(changes) hash set, and sorts only the selected compositor
entries back into display-list order. It does not allocate or clear an array
sized to all historical entries. Composition writes an exact R8 damage mask and
draws each selected cached source once through that mask; disjoint repair
regions therefore do not create a regions-times-sources loop. Moving one item
in a 64-boundary grid selects one source, not 64; a quiet grid performs no paint,
candidate planning, or composition.

Empty paint runs do not allocate textures. A parent with ordinary paint both
below and above a boundary pays exactly two additional tightly bounded run
surfaces. The memory tax is explicit: two RGBA textures plus one R8 damage mask,
nine bytes per pixel of each run's rectangular allocation, in addition to the
parent presentation and boundary surfaces.

Immediate-mode third-party paint has no retained identity, coverage, or stable
ordering contract, so it cannot be placed in an ordered cached run. A target
containing such paint uses the same flat retained-surface representation as a
target without boundaries while that paint exists. It remains volatile and
fully correct; boundary placement on that target is not compositor-only. When
the last volatile writer disappears, its old coverage is invalidated and the
target atomically returns to ordered sources. Implementing
`RetainedUiMaterial` supplies the missing contract and keeps ordered
composition active. GPU tests cover both the coexistence and departure
transitions.

Paint routed into a boundary must also use the boundary's local coordinate
space. Stock `UiMaterial` extraction now derives that space from the declared
paint target and localizes both its transform and clip. The ordered compositor
exposed the previous mismatch because the baked path happened to draw the
global transform into its parent. The nested unretained-material test proves
the local target contract independently of retention support.

The flat presentation cache is deliberate. Replaying every source directly
into the window would avoid updating that cache during motion, but would make a
static UI pay one draw per source forever. Conversely, arbitrary interleaving
cannot be reconstructed from one flat ordinary-paint texture. Ordered retained
sources plus damage composition are the only representation here that keeps
both arbitrary order and the one-texture quiet path.

### Does ordered-source management pay for itself?

It is not free, and the boundary count is intentionally an API-level resource
decision rather than an inferred promotion heuristic. With `B` declared
boundaries, the renderer owns `B` boundary surfaces plus at most `B + 1`
ordinary runs, their canonical boundary records, and ECS change-detection
scans. A boundary should therefore describe a stable subtree whose placement
can move independently, not decorate every leaf.

The steady and changed contracts are different:

- without a boundary, none of the ordered-source structures or passes exist;
- with quiet boundaries, extraction still pays Bevy's `Changed<T>` archetype
  scans, but the render world performs no candidate plan, paint pass, or
  composition pass;
- one change queries the persistent display-list index, stores only actual
  hits/direct changes, sorts only those hits, and submits every selected source
  once through the damage mask;
- source allocation and payload bytes, exact resize copies, paint pixels,
  composition pixels, admitted source-scissor pixels, and unique source
  submissions have separate counters.

The stress executable accepts `--layout-group 1 --repaint-boundaries`, so the
deliberately excessive case of 10,000 one-node boundaries is reproducible
beside the intended 10/100/1,000-node subtree sizes. An unoptimized diagnostic
run on 2026-08-14 found that moving one boundary added about 0.08 ms over the
otherwise identical quiet 10,000-boundary topology, while the topology itself
was very expensive in debug builds. That number is not an optimized baseline;
it is evidence that the changed planner no longer walks or clears all 10,000
entries. The fresh optimized and physical-device sweeps remain required before
choosing a product boundary granularity.

"Zero paint" must not be reported as "zero output pixels." Moving a visible
boundary changes the parent's old and new pixels, so those pixels must either be
composed into the presentation cache or regenerated in the final output every
frame. Counters and tests distinguish paint repairs from cached-source
composition. The placement contract is zero extracted child paint, zero child
raster, and exact old/new compositor damage; it is not zero GPU writes.

Boundary content damage is mapped from source pixels through the boundary's
affine UV transform into exact conservative parent rectangles. A scaled proof
changes 336 source pixels, repairs those 336 pixels in the child and exactly
1,344 mapped pixels in the parent; it does not promote the change to the full
boundary. Propagation records the exact source damage epoch, not a boolean dirty
bit, so a source that repairs and changes again on the next frame propagates the
new generation instead of becoming permanently stale. Boundary transform and
group opacity live only on the parent display record. Changing either damages
old union new parent coverage and never dirties the child surface, layout,
geometry, stack, or clipping.

Zero opacity is an exact suspension state. Content changes still update
canonical records, but their damage is acknowledged without raster or parent
work. An initially hidden source allocates no texture, and hiding a previously
visible boundary retires that boundary's own surface. Cached descendant
surfaces may remain allocated while an outer boundary is hidden, but perform no
work. Effective visibility walks the declared boundary ancestry, so a hidden
outer boundary also suspends nested sources. Revealing it invalidates every
visible descendant source and repairs them deepest first before the parent
composition; GPU tests cover both the zero-work hidden interval and the
complete reveal.

The cache uses the parent's render-target format. With the common four-byte
RGBA8 target, an intermediate surface adds one quantization point: direct and
one-boundary alpha composition can differ by at most one stored channel value
in the acceptance scenes, and nesting can accumulate one value per cache
level. This bounded precision cost is tested explicitly. Forcing an eight-byte
half-float cache for an RGBA8 parent would reduce it but double retained memory
and compositor sampling bandwidth; preserving the parent format is the
performance default and also avoids an implicit format conversion policy.

## Handoff audit

The supplied prior implementation report is evidence, not specification.

The local `bevy-ui-retain` branch was audited as another evidence source. It
contains useful stable owner/arena allocation work, but nominates correctness
through a long explicit `Changed<T>` and `RemovedComponents<T>` inventory and
does not retain final pixels. Its code is not used as the new foundation. The
post-0.19 upstream PR stack is also not cherry-picked because it includes broad
render API churn beyond UI. The implementation keeps the compatible idea—a
stable retained allocation—while making candidate nomination subordinate to
canonical paint-record comparison.

Retained as invariants:

- premultiplied layer composition;
- old union new coverage for moved, removed, or reordered content;
- wipe then rebuild damaged pixels bottom-up;
- stable identity independent of transient list indices;
- damage remains owed until a complete repair is encoded;
- offscreen readers are explicit dependencies;
- text coverage composes glyph-local translation with node transforms;
- GPU readback, red-first regression tests, and retained-vs-full differential
  tests are the correctness instruments;
- declared repaint boundaries for content whose placement animates while its
  pixels remain stable.

Rejected from the initial design:

- a `0.5px` movement threshold;
- a fixed three-frame consumer settling period;
- a periodic heartbeat repaint;
- correctness based on a handwritten `Changed<T>` list alone;
- region-count, area, or promotion thresholds that create behavior cliffs;
- clearing dirty GPU uploads before the upload path is known to be available.

Still requiring measurement rather than assumption:

- fullscreen texture versus sparse/tiled surface storage on each target GPU;
- mobile tile-memory load/store behavior for the exact render-pass shape;
- whether folding UI composition into Bevy's final pass materially changes
  power consumption;
- the best deterministic tile size, if tiled storage is implemented.

## Verification contract

Every correctness feature lands with an automated proof where the platform API
allows it:

- unit tests for canonical rectangle unions, coverage mapping, stable identity,
  dependency propagation, order changes, removals, and owed damage;
- schedule tests proving static systems and extraction do not run;
- GPU readback tests for blending, wipe/repair, old/new movement, clipping,
  text transforms, sampled surfaces, resize, resource unavailability, and every
  presented state of repeated animations;
- a deterministic adversarial scene rendered once with retention and once with
  full redraw, followed by an exact pixel comparison;
- counters asserted by tests: entities extracted, paint records changed,
  canonical records staged, items and prepared quads replayed, damaged pixels,
  paint repairs, retained texture bytes, cached-source composition, final
  presentations, and physical UI texture samples.

Every regression test must first fail with the named defect reintroduced.
The production GPU harness also builds both `ExtractSchedule` and `Render` with
ambiguity detection at `Error`. Enabling it found and fixed a real unordered
shared-resource conflict between viewport and image/text dependency extraction;
the backstop remains active in every GPU case.

Performance has two separate test layers. Large-scene deterministic contracts
run with the normal test suite and assert exact work cardinality, so timing
noise cannot hide an asymptotic regression. Criterion measures constant factors
and supports named baselines through Bevy's existing
`cargo bench -p benches --bench ui -- --save-baseline <baseline>` workflow. The
retained renderer adds the first dedicated `ui` benchmark target in this tree.

The current matrix is exact about what it includes:

| Layer | Sizes | Shapes | Mutations | Deliberately excluded |
|---|---:|---|---|---|
| `Changed<T>` nomination | 100, 1,000, 10,000 entities | one archetype | quiet, one changed, all changed; one input and an eight-input `Or` | canonical extraction and rendering |
| Main-world UI work | 100, 1,000, 10,000 nodes | flat, four-way balanced, 100-node independent roots, explicit containment groups of 10/100/1,000; a 256-deep chain | quiet, equal `Node` write, one/all layout, one local node-geometry change, one placement, one/all compositor-boundary placement, one change per layout boundary, all contained leaves; one reparent for the forest; 64 consecutive contained-layout frames | render extraction and GPU work |
| Canonical paint and exact damage | 100, 1,000, 10,000 records | adjacent tiles, separated pixels, full overlap | quiet, one paint change, all paint changes; indexed intersection and area against 10,000 separated regions | Bevy extraction and rasterization |
| GPU acceptance | small 64-by-64 scenes, including 64 independent boundaries | disjoint and translucent overlap across every integrated family; arbitrary and nested repaint boundaries; ordinary paint on both sides of a boundary | quiet, targeted run/content changes, run growth with preserved siblings, one-of-many compositor placement, transform/opacity, reflow, removal, immediate-mode fallback transitions | stable wall-clock timing |
| Windowed stress executable | configurable, 10,000 by default | grid, full overlap, alternating overlap; background, text, image, gradients/shadows/borders, or mixed; optional independent repaint groups | quiet, one/all paint, one/all placement, one/all layout, one/all boundary placement, one churn | automated pass/fail timing thresholds |

Layout topology and paint overlap are orthogonal inputs: overlap does not alter
Taffy's dependency graph, so the layout Criterion cases do not duplicate every
paint geometry. The windowed stress executable composes them, allowing (for
example) 10,000 fully overlapping nodes, 100-node layout containment, mixed
paint families, and one or all layout animations in the same run.

The deterministic 10,000-record tests prove that quiet retained paint submits
zero candidates, one animation submits one candidate, 10,000 animations submit
exactly 10,000 candidates, and one remove/reinsert performs constant record
work. Exact damage-index tests compare every indexed result with exhaustive
intersection and area calculations, including a 10,000-region separated case.
GPU tests separately assert staged records, records and quads replayed, damaged
pixels, paint repairs, cached-source composition, final presentations, and
physical UI texture samples.
Main-world counters prove that quiet and paint-only frames execute no Taffy,
recursive geometry, stack, or clipping walk. Layout-scope unit tests assert exact Taffy
computation and geometry-visit counts for multiple roots, nested containment,
marker insertion/removal, reparenting between boundaries, target-scale changes,
and ghost-node flattening. A repeated-frame contract animates one leaf for 64
consecutive frames and requires exactly one Taffy computation and nine geometry
visits on every frame; an unrelated subtree may never enter the work set. Grid,
intrinsic measurement, percentage sizing, absolute placement, and hidden nodes
are compared directly against the same tree without containment. These tests do
not pretend the remaining change-detection scans cost zero.

- [Bevy benchmark instructions](../benches/README.md)

Short Criterion samples on the development machine on 2026-08-11 produced the
following local comparison points. They used a 100 ms warmup, 200 ms target
measurement, and ten samples; they are smoke measurements, not portable
baselines:

| 10,000-node shape and mutation | Stock | Retained |
|---|---:|---:|
| flat, quiet | 0.461 ms | 0.208 ms |
| flat, one equal `Node` write | 0.426 ms | 0.242 ms |
| flat, one radius change | 0.400 ms | 0.252 ms |
| flat, one genuinely coupled layout change | 4.171 ms | 4.100 ms |
| flat, all layout changes | 5.816 ms | 5.637 ms |
| contained 10, one internal layout change | 0.307 ms | 0.202 ms |
| contained 100, one internal layout change | 0.296 ms | 0.173 ms |
| contained 1,000, one internal layout change | 0.511 ms | 0.440 ms |
| contained 100, boundary's own layout changes | 0.805 ms | 0.686 ms |
| contained 100, one internal change in every boundary | 2.868 ms | 2.778 ms |
| contained 100, all internal leaves change | 4.453 ms | 4.236 ms |

The repaint-boundary increment adds `one_boundary_compositor_change` and
`all_boundary_compositor_changes` to every contained Criterion shape. In the
10,000-node absolute-contained-100 scene, changing one boundary property in
`PostUpdate` measured 0.134 ms retained versus 0.274 ms for the equivalent stock
`UiTransform` subtree placement. The retained main-world counters stayed at one
initial layout, geometry, stack, and clip run throughout the animation.

The 1280-by-720 windowed stress executable was also run for 240 frames with 120
warmup frames on the development RTX 4090/Vulkan machine on 2026-08-12. These
are local end-to-end samples, not cross-device claims:

| Sustained 10,000-node placement | Mean | p95 | Max | Frames >= 4 ms | Longest >= 4 ms streak |
|---|---:|---:|---:|---:|---:|
| stock, one transform containing 10,000 nodes | 3.430 ms | 3.776 ms | 4.327 ms | 1 | 1 |
| retained, one repaint boundary containing 10,000 nodes | 1.219 ms | 1.348 ms | 1.506 ms | 0 | 0 |
| stock, 100 transforms each containing 100 nodes | 3.212 ms | 3.583 ms | 4.530 ms | 3 | 2 |
| retained, 100 repaint boundaries each containing 100 nodes | 1.564 ms | 1.693 ms | 2.559 ms | 0 | 0 |

The retained boundary sources allocate each surface once. Thereafter boundary motion
changes only 1 or 100 parent display records per frame; none of the 10,000 child
paint records are compared or rerasterized. The parent still repairs the exact
old/new composite coverage and the final fused blit still samples the visible UI
layer, so these numbers do not mislabel composition as free.

Each Criterion iteration mutates the next state and immediately runs
`PostUpdate`, so every change row is sustained consecutive-frame work rather
than a single cold spike. Consequently, a width animation inside one genuinely
coupled 10,000-node flex root can cost about 4.1 ms on every animation frame on
this machine; changing all 10,000 widths can sustain about 5.6 ms. In this
isolated main-world benchmark retention does not regress either case, but it
cannot skip a real Taffy dependency. The end-to-end table below separately
includes render-world cost. Containment changes the dependency: one changed leaf
inside a 100-node widget measured 0.173 ms retained, one changed leaf in every
100-node widget measured 2.778 ms, and changing every leaf measured 4.236 ms.
Independent dirty widgets therefore add; containment bounds the term but does
not make an actually changing whole interface free.

The contained results show the intended scaling law: a mutation pays for the
changed candidate scan plus the smallest semantically coupled widget, not the
whole menu. Changing one leaf in every boundary or every contained leaf remains
an intentionally expensive case because it genuinely invalidates all groups.
The benchmark includes both cases rather than presenting containment only in
its best case.

A 256-deep chain previously measured 37.9 microseconds retained versus 84.9
microseconds stock when quiet, 49.6 versus 83.9 microseconds for one placement
change, and 291 microseconds for one layout change on both paths. A 1,000-deep
chain overflowed a Bevy task-pool thread's stack during initial layout, so it is
recorded as an unsupported stress result rather than silently omitted or
reported as a timing.

At 10,000 entities, a quiet one-input `Changed<T>` query measured about 9.13
microseconds and the eight-input `Or` about 29.13 microseconds. One changed
entity measured 8.97 and 28.31 microseconds respectively; returning all 10,000
measured 17.99 and 24.23 microseconds. These scans explain much of the retained
quiet path's remaining size dependence.

Canonical paint repair planning measured about 6.43 ns when quiet and 0.246
microseconds for one change, independent of whether 100 or 10,000 records are
retained. After caching repeated unions, changing all 10,000 adjacent records
measured about 90 microseconds and resizing all 10,000 about 130 microseconds in
the local Criterion run. Separate benchmarks retain the scattered and complete-
overlap shapes. Equal old/new coverage is journaled once, not duplicated as
separate old and new damage events. The common zero/one-region coverage forms
allocate nothing.

The former sorted-phase replay benchmark was deleted when production stopped
using that algorithm. Canonical records are now filtered through a persistent
exact two-dimensional spatial index before entering transient queue and prepare
buffers. A one-item repair therefore stages intersecting records rather than
building a 10,000-item phase and culling it at draw time. Directly changed
groups bypass the query, dense direct changes bypass tree refitting, and partial
queries refit every outstanding bound first. The index has no tile-size or
count threshold and is exhaustively differential-tested.

The GPU acceptance harness uses a 64-by-64 image target, synchronous pipeline
compilation, explicit device polling, an in-process mutex, and a cross-process
file lock. Its stock and retained apps run in one process. A directly
constructed stock final scene is byte-identical to the retained scene reached
through movement, recoloring, and removal, including translucent overlap.

The integrated retained families are box shadows, backgrounds
(`BackgroundColor` and `OuterColor`), ordinary, sliced, and tiled `ImageNode`s,
camera-backed `ViewportNode`s, background and border gradients, solid borders
and outlines, all stock UI text paint, and opted-in custom `UiMaterial`s. They
share one scene, one stable
`(entity, family, ordinal)` identity space, and one damage journal per camera.
Family extractors only update canonical records; a single later replay stage
sorts every visible family together. This is required for correctness:
repairing a translucent image or glyph must first replay the background
beneath it. Ordinary nodes, images, glyphs, gradients, and shadows retain their
prepared instance data and use mask-aware instanced pipelines. Slices and
arbitrary materials reuse Bevy's pipelines with exact-scissor fallback. The
named GPU image proof changes a 10-by-10 image, repairs exactly 100 pixels, and
replays exactly the two intersecting logical items bottom-up.

All image-reading families share one exact reverse-dependency graph. Access is
an explicit locked extraction transaction rather than an ECS `ResMut`: generic
material plugins cannot declare a static order relative to other
monomorphizations of the same extractor. The GPU material tests register two
retained material types under strict schedule ambiguity detection, so adding a
new material type cannot turn a valid app into a launch-order lottery.

An ordinary `UiMaterialPlugin` remains correct without opting into the exact
contract. Its extractor declares the actual camera or repaint-boundary surface
volatile once per target, not once per node; planning then journals a full
target repair through the same owed-damage path. A GPU differential places such
a material inside a translated boundary and proves that it is rasterized into
the child surface before parent composition. Opting into
`RetainedUiMaterialPlugin` replaces that conservative cost with exact keys,
coverage, and image dependencies.

Records use `Changed<T>` candidate nomination, bit-exact canonical values,
stable render entities, and old-union-new damage. The exact, non-overlapping
damage union is wiped to transparent and rebuilt from only the sorted items
whose physical bounds intersect it. A localized background proof moves a
10-by-10 leaf across a larger background: exactly 200 pixels are repaired and
the two necessary logical items are each replayed once through the exact mask.

Placement is extracted once per changed `UiGlobalTransform`, before the paint
families. Every retained draw stores its entity-local translation, so that one
transaction updates all of the entity's canonical transforms, clips, prepared
instances, and exact old/new coverage. Color, image, gradient, border, shadow,
material, and text extractors no longer re-read their style data for a pure
placement change. Storing the local offset directly, instead of recovering it
with a matrix inverse, makes a zero-scale-to-visible transition exact. Painted
records clipped to empty coverage stay canonical but out of the spatial/order
index, allowing placement alone to reveal them again. GPU differentials cover
mixed-family translation, fully clipped re-entry, and zero-scale recovery.

Declared `RepaintBoundary` placement is a different domain from ordinary
`UiTransform` placement. Its `transform` and `opacity` are extracted once into
the parent surface's boundary record. A 28-by-24 translation proof changes one
canonical record, repairs one parent surface and the exact 1,104-pixel old/new
union, replays only the underlying parent item plus the boundary quad, and
allocates or repairs no child surface. An opacity-only proof has the same
one-record/one-parent-repair contract. Identity, arbitrary sibling ordering,
nested boundaries, paint clipping, post-flatten group opacity, removal and
surface-memory release all have GPU readback proofs.

The mutation audit found that `ComputedNode` alone is not a complete
nomination source: Bevy intentionally writes resolved borders and corner radii
through `bypass_change_detection()`. `Changed<Node>` and
`Changed<ComputedUiRenderTargetInfo>` therefore nominate those derived paint
inputs; canonical comparison still decides whether the record and pixels
actually changed. A red-first GPU test changes only the source border while
the node size and transform remain fixed. Retargeting to an entity without a
render camera also removes the old camera's records immediately instead of
leaving stale ownership behind.

The current background dependency matrix is executable in `tests/gpu_ui.rs`:

| Pixel dependency | Nomination or lifecycle source | Named proof |
| --- | --- | --- |
| fill color | `Changed`/removed `BackgroundColor` | equal write, real color change, adversarial despawn |
| outer rounded-corner fill | `Changed`/removed `OuterColor` | `removing_outer_color_repairs_its_vacated_pixels` |
| resolved size/border/radius | `Changed<ComputedNode>`, plus source `Node` and target info | movement and `source_node_changes_nominate_bypassed_computed_paint_geometry` |
| target-space transform | `Changed<UiGlobalTransform>` | `ui_transform_motion_nominates_only_the_moved_leaf` |
| inherited visibility | `Changed`/removed `InheritedVisibility` | `inherited_visibility_removes_only_the_hidden_leaf_pixels` |
| calculated clip | changed, inserted, or removed `CalculatedClip` | named clip insertion and removal tests |
| target camera ownership | `Changed`/removed `ComputedUiTargetCamera` | `losing_a_renderable_target_removes_pixels_from_the_previous_camera` |
| paint order | `Changed`/removed `ComputedStackIndex` | `computed_stack_changes_repair_translucent_paint_order` |
| UI participation | changed/removed source `Node` | `removing_node_ends_ui_participation_even_if_computed_components_remain` |
| viewport size and scale | `Changed<ComputedUiRenderTargetInfo>` plus surface mismatch | `resizing_a_viewport_reconstructs_its_retained_surface` |

All canonical draw fields remain in the fingerprint, so these sources only
nominate comparison. They do not decide damage.

Image and font-atlas resource dependencies use one exact reverse index. Each
visible reader declares the image texels it can sample. An asset event compares
those CPU-side bytes before nominating the reader, so equal writes or changes
outside a selected `ImageNode` rectangle cause zero extraction, canonical
comparison, or repair. Full-image readers keep one exact byte snapshot per
shared sample rather than one copy per node. Transparent, invisible, empty,
and built-in-transparent nodes do not subscribe; their own component changes
re-establish dependencies if they become paintable.

Rectangle precision has a proof boundary. Single-mip, two-dimensional,
single-layer images with clamp-to-edge, non-comparison, non-anisotropic
sampling track the selected rectangle plus one texel of nearest/linear filter
reach. Clamped rectangles wholly outside the image still track the edge texel
they actually sample. Repeating or mirrored addressing, mipmaps, array layers,
anisotropy, and unsupported CPU pixel layouts conservatively track the whole
image. This is a correctness fallback, not an admission heuristic.
`TextureAtlasLayout` changes separately nominate only nodes using that layout,
then canonical comparison decides whether the selected rectangle changed. The
GPU suite proves quiet images match stock output, image tint changes rebuild
underlying backgrounds, sampled pixel modifications repaint exact readers,
unsampled and equal-byte modifications do no retained work, selected atlas
changes repaint, and irrelevant atlas edits do not.

Sliced and tiled nodes are canonical records in that same image identity;
switching modes does not create a parallel retained path. Their fingerprints
include target and atlas rectangles, tint, flips, inverse scale, slice borders,
scale modes, and every floating-point mode parameter as exact bits. They reuse
the common sampled-image dependencies. GPU differentials cover quiet sliced
and tiled output, tint repair, and same-texture siblings; the last asserts that
a one-node repair submits exactly one prepared quad rather than the stock
texture batch.

Camera render targets expose a dependency that CPU image bytes cannot capture.
Every active camera using `CameraOutputMode::Write` makes its image target
explicitly volatile; each frame nominates only readers of that image. A camera
using `Skip` provably does not write the final target and is excluded. The GPU
suite changes an active camera clear color and proves both `ViewportNode` and an
ordinary `ImageNode` update; deliberately removing target marking leaves the
old color behind and makes the differential fail. When the source camera is
inactive, a quiet viewport has zero further comparisons or repairs. Active
targets with no retained readers do not even advance a resource revision. CPU
image changes, source-target switches, equal replacement, and removal each
have separate proofs; localized viewport changes repair exactly its 20-by-20
box.

Custom render and compute writers use the public, thread-safe
`RetainedUiImageWrites` render-world resource. `invalidate` declares a full
image write; `invalidate_region` accepts an exact physical-texel rectangle and
nominates only samples whose filter-expanded read regions intersect it.
Invalid rectangles safely become full-image invalidations, while empty ones do
nothing. A unit proof has two readers of one image and nominates only the reader
intersecting a declared GPU write.

Background and border gradients retain one canonical record per list entry.
Resolved geometry, interpolation color space, exact-bit stops and hints,
border geometry, order, target scale, transform, and clip all participate in
comparison. Empty and fully transparent gradients create no records. Border
gradient coverage is the exact union of its four rounded edge reaches, not the
node box: a 24-by-20 test node repairs 340 possible border pixels and never its
140-pixel center. GPU differentials cover linear, radial, conic, solid,
mixed-stack ordering, border mutation, equal replacement, and removal. The
prepared-quad counter also proves that a three-stop change submits its two
gradient segments and no unrelated item.

Each declared box shadow has one canonical record containing its resolved
target-space size, blur reach, corner radii, color, order, transform, clip, and
camera sample count. Coverage is the shader's full possible output box, clipped
to the target. A color change repairs exactly 1,296 possible pixels and submits
one quad; a four-pixel offset repairs the exact 1,440-pixel old-union-new
region. Removal, multiple-shadow back-to-front order, equal replacement, and
quiet-frame behavior have GPU proofs. Another proof changes a shadow beneath a
translucent background and requires both commands to replay bottom-up, while a
transparent shadow creates no retained surface. `BoxShadowSamples` is a camera
input, not a node input: changing it nominates only nodes targeting that camera
and its value participates in canonical comparison. Deliberately removing that
reverse nomination makes the pixel differential fail, confirming that the test
detects the stale-sampling defect rather than passing vacuously.

Availability follows Bevy's render-asset lifecycle rather than main-world
`Assets<Image>` membership. Removing a main-world asset does not unload its GPU
copy while a strong handle remains, so it correctly causes no repaint. A
never-available image is safely omitted and its old coverage erased. An added
or relevantly modified image remains pending until `RenderAssets<GpuImage>`
contains the new revision; while a byte-upload budget deliberately holds it
back, the old retained pixels stay visible and the damage remains owed.
Per-item batch components are cleared before preparation so readiness can never
be satisfied by stale metadata from a previous frame.

Custom materials use an explicit correctness contract because arbitrary WGSL
cannot be invalidated safely by inspecting an asset handle. `RetainedUiMaterial`
declares three things:

- an exact, collision-free key containing every non-image shader input, or
  `Volatile` when globals, time, or other per-frame state may affect output;
- `Node` coverage only when the vertex shader cannot rasterize outside the
  transformed node quad, otherwise conservative `Target` coverage;
- every sampled image, optionally narrowed to a physical-texel read rectangle.

The renderer canonicalizes and deduplicates image declarations. The complete
declaration, not merely the paint key, participates in asset revision and
prepared-bind-group readiness. Sample bytes and metadata use the same exact
reverse dependency index as built-in images, so an equal image write does no
work and a relevant write nominates only its material readers. If a new image
binding is not GPU-ready, old pixels remain visible and damage remains owed.
Ten focused GPU tests cover exact quiet paint, equal and real asset writes,
sample mutation, binding replacement and unavailability, removal, explicit
volatility, conservative target coverage, and the full-repaint fallback for an
ordinary `UiMaterialPlugin`.

The engine can test that it obeys a declaration, but cannot prove that an
arbitrary shader declaration is truthful: WGSL reflection cannot determine
whether a uniform contains time, how a custom vertex shader expands coverage,
or which dynamically indexed texels can affect which output pixels. That is a
real API proof boundary. The safe declarations are `Volatile`, `Target`, and a
full-image sample; narrower promises are application contracts and should have
retained-versus-stock GPU differentials for each custom shader. A changed texel
inside a declared material sample currently invalidates the material's entire
declared output coverage because an arbitrary shader may broadcast that texel
everywhere. A future output-region mapping API is justified only when an
application can prove a narrower influence relation.

Solid borders and outlines each retain one compound record with four canonical
edge colors and four exact coverage regions. Persistent preparation groups
equal colors by OR-ing their edge flags, matching stock's single-blend corner
ties without retaining four arena records. Damage comparison remains per edge:
changing the left edge out of an equal-color group damages only that edge's
140-pixel rounded-corner reach, while the other three retain their pixels.
Equal-color borders, distinct-color regrouping, outlines, and component removal
are byte-identical to stock in named GPU tests.

Text retains consecutive glyphs in the same section that use the same
font-atlas texture as one draw record. Identity uses the stable section entity
and a section-local atlas-run ordinal; paint order is separate canonical data.
A child-span color change therefore changes exactly that span's glyph record
instead of churning every later run, while overlapping spans still paint in
layout order. The canonical value contains the bit-exact glyph colors, local
translations, atlas rectangles, node transform, clip, order, and target;
coverage composes each glyph-local translation with the node transform rather
than incorrectly measuring glyphs around the origin. A content change damages
the exact old and new glyph unions, so shortening a string wipes the vacated
glyphs without repainting its full layout box. Root colors and child
`TextSpan` colors have separate nomination paths: a reverse section-to-root
index makes a child-only color mutation rebuild the owning glyph run.

Font-atlas invalidation uses that same sampled-image index. Each retained run
records the atlas cells it can sample, including the one-texel bilinear reach
around each glyph. Appending an unrelated glyph or changing any other
unsampled atlas pixel therefore produces zero text candidates, canonical
changes, or repair work; changing a sampled pixel advances only its readers.
Texture and sampler metadata that can alter sampling is compared separately,
with debug labels and creation-only flags normalized away. Dependency updates
are transactional so temporarily detaching a root during re-extraction cannot
discard a just-observed revision. GPU tests cover sampled and unsampled
mutations, span-only color changes, vacated glyphs, and delayed font-atlas
uploads. A delayed upload keeps the old text visible and damage owed until the
new `GpuImage` is actually ready.

All stock text paint families are now first-class retained records: run
backgrounds, glyph and decoration shadows, underline and strikethrough,
selection backgrounds, selected glyph color, cursor, and IME preedit
underlines. `EditableText` is a peer root rather than a `Text` subtype, so the
root query explicitly accepts either component. Section decoration components
use the same reverse section-to-root dependency index as span colors. Input
focus is retained as explicit state: a focus transition nominates only the old
and new focused entities, and canonical comparison limits damage to selection
pixels whose focused/unfocused color actually differs. GPU differentials cover
quiet and mutated shadows, all run decorations, decoration removal, selected
editable text, focus-only selection recoloring, and nonempty IME underline
geometry.

Transparent or empty paint is represented by no retained record. This matters
because `Node` requires transparent background and border components: keeping
those as invisible records would still compare and rewrite canonical state
whenever an unrelated transform moved. A GPU/counter proof moves an unpainted
node and observes zero paint candidates, records, surfaces, or repairs.

A paint record owns an exact set of physical coverage rectangles rather than
one bounding box. Its compact `Empty | One | Many` representation allocates
nothing for ordinary records; genuinely disjoint sampled-image mappings pay
for storage only when they use it. Damage and item intersection consume the
same region set. A unit proof keeps the gap between two regions absent from
damage; composition no longer needs a second coverage representation.

The owned UI layer is double-buffered and owns one byte-per-pixel damage mask.
A repair encodes into the inactive texture and replaces the visible texture
only after every region succeeds.
Each slot carries a committed generation; bringing an older slot current
copies only damage committed since that generation, rather than copying the
whole layer. Those exact rectangles are copied by one instanced GPU draw; an
early implementation emitted one texture-copy command per rectangle and the
10,000-separated-region stress case exposed that command-count cliff. The same
instance buffer wipes every current damage rectangle before repaint. Failed or
not-yet-compiled work remains owed. On a quiet
background-only frame, GPU counters prove zero additional repairs, repair
pixels, and replayed items while the final presentation sample continues. An equal component
replacement nominates and compares exactly one record but changes zero records
and causes zero repairs; a real color change repairs exactly that record's old
and new physical coverage.

Only records intersecting exact damage are staged. Renderer-owned families are
grouped into persistent instanced runs and draw once through the mask even when
damage is disjoint. Two disjoint changed quads are proven to stage two records,
repair 200 pixels, report two logical items, and draw two quads from one run.
There is no node-count, damaged-area, or region-count threshold.

Four readback tests inspect animation streams rather than only final frames. One
moves a translucent item through three disjoint positions and requires at least
24 captured frames to cycle through three stock-rendered complete states with
no stale repeat, ghost, partial repair, or skipped state. It was confirmed red
by removing inactive-slot synchronization. The other cycles a translucent
full-surface repair through three states on the batched path and enforces the
same generation ordering; it was confirmed red by removing the wipe, which
immediately exposed alpha accumulation. A third moves an unchanged repaint
boundary through three compositor positions and accepts only complete cached
states. The fourth changes content inside a static boundary every frame and
requires each child and parent surface generation to land atomically. It first
failed because a boolean "already propagated" marker suppressed every source
epoch after the first, then exposed the shared-rectangle-buffer overwrite that
mixed two generations within one box. These tests exercise Bevy's pipelined
render app against an image target, not a window-system compositor.

Each layer is sized in viewport-local physical pixels. UI repair uses that
local surface directly; only composition applies the camera viewport. A
nonzero-origin viewport is byte-identical to stock, and moving an unchanged
viewport is proven to cause zero paint candidates and zero repair work. This
is placement-only motion of a retained surface.

Composition is folded into Bevy's existing final blit. A quiet visible layer
therefore causes no extra pass, attachment load/store, or draw, but it does add
one retained-layer texture read across the physical viewport. A 10-by-10 UI in
the 64-by-64 harness truthfully reports 4,096 sampled UI pixels, not 100 logical
content pixels. Two disjoint translucent regions remain byte-identical to stock
and preserve the untouched gap without retaining or rebuilding a coverage union
that the fused shader cannot use. An empty committed layer uses the transparent
fallback, and a camera with no visible UI allocates no layer textures. If a
layer is lost or no longer matches its viewport size or target format while its
records remain unchanged, the whole viewport is added to owed damage and those
records replay on the following extraction.

The windowed stress matrix is intended to run unchanged on the target device:

```text
cargo run --profile stress-test -p bevy_ui_render_retained --features stress_test \
  --example stress_test -- \
  --renderer retained --geometry overlap --family mixed \
  --workload one-layout --nodes 10000 --layout-group 100 \
  --frames 1200 --warmup 120
```

Run the same command with `--renderer stock` for the A/B. Geometry accepts
`grid`, `overlap`, and `alternating`; family accepts `background`, `text`,
`image`, `effects`, and `mixed`; workload accepts `quiet`, `one-paint`, `all-paint`,
`one-placement`, `all-placement`, `one-layout`, `all-layout`,
`one-boundary-placement`, `all-boundary-placement`, and `one-churn`. Boundary
placement requires grid geometry plus `--layout-group` and
`--repaint-boundaries`; retained mode moves `RepaintBoundary::transform`, while
stock moves the equivalent group `UiTransform`. The repaint flag is independent
of workload, so the same boundary topology can be measured while quiet, while
one boundary moves, and while every boundary moves.
Omitting `--layout-group` creates one coupled root; supplying a positive size
partitions the nodes into explicit `LayoutContainment` widgets of that size.
The 10/100/1,000 sweep used by Criterion can therefore be repeated on physical
target hardware without changing code.
Paint animation deliberately alternates two nearby slate colors. The values are
bit-distinct and exercise the same changed-record path without turning an
unattended performance run into a visually ambiguous full-window strobe.
An unbounded run reports retained work counters every 120 frames. A finite run
does no periodic logging during its measured interval; after `--warmup` frames
it records every full-frame duration and finally prints the mean, p50, p95,
p99, maximum, number of frames at or above 4 ms, and longest consecutive run at
or above 4 ms. This distinguishes a single setup spike from a sustained busy-UI
failure without making log output create the tail it is measuring. The retained
counter report includes the current retained texture payload bytes; that gauge
is exact for the two RGBA layer textures plus the R8 damage mask, but excludes
driver bookkeeping. This
executable has no universal pass/fail frame-time threshold:
the same command is the measurement instrument, while the acceptable budget is
chosen for the game's target hardware and frame rate.

The same development machine's fat-LTO `stress-test` Vulkan runs on 2026-08-12
measured the flat-parent implementation immediately before ordered source runs.
They are retained as historical comparison data, not current compositor
results. Every row contains 120 consecutive measured frames after 60 warmup
frames and used the same executable for stock and retained:

| 10,000-node end-to-end windowed workload | Stock | Retained |
|---|---:|---:|
| grid mixed, quiet | 17.220 ms | 1.013 ms |
| grid mixed, one paint change per frame | 17.730 ms | 1.250 ms |
| overlap background, one paint change per frame | 3.302 ms | 1.488 ms |
| overlap background, all paint changes per frame | 3.147 ms | 3.469 ms |
| grid effects, all paint changes per frame | 44.549 ms | 34.683 ms |
| grid mixed, all paint changes per frame | 17.410 ms | 16.488 ms |
| grid background, all ordinary placements change | 3.380 ms | 3.020 ms |
| grid mixed, all ordinary placements change | 17.211 ms | 21.727 ms |
| grid background, all widths change | 4.942 ms | 6.622 ms |
| grid effects, all widths change | 45.179 ms | 45.177 ms |
| grid text, all widths change | 168.475 ms | 170.926 ms |
| grid mixed, all widths change | 48.427 ms | 54.242 ms |
| grid mixed, one layout change in a 100-node containment | 17.513 ms | 1.162 ms |
| grid mixed, all 100 declared repaint boundaries move | 17.754 ms | 1.655 ms |

The quiet, localized paint, contained layout, and declared-boundary placement
retained rows had no frame at or above 4 ms in that implementation. The
overlapping one-change row is
deliberately adversarial: one changed translucent node intersects all 10,000
contributors, yet retained still wins because unchanged canonical preparation
remains resident. Effect-heavy and mixed scenes also win under complete paint
mutation because their persistent batching outweighs proof cost.

Global mutation is the honest boundary, and this table deliberately retains
the non-winning rows. Changing all 10,000 overlapping simple backgrounds is
about 0.32 ms slower; a reverse-order confirmation also lost (4.259 versus
3.028 ms, with a noisier retained tail). Changing all mixed widths is about
5.82 ms slower end to end, confirmed in reverse order at 54.266 versus 48.412
ms. The main-world portions were nearly identical (46.463 versus 46.127 ms);
the remainder is canonical coverage proof, retained replay, surface repair,
and composition. The text-only result is similarly dominated by genuine Bevy
reflow: retained proved that final glyph pixels were unchanged and did no
raster repairs after initialization, but total time still measured 170.926
versus 168.475 ms.

These are not bugs that an area threshold or inferred mode switch can solve.
Immediate mode only produces the new frame when every item changes; exact
retention must first prove which old pixels are invalid. A claim that one path
is strictly faster for every possible mutation distribution is therefore
false unless the implementation keeps a second immediate renderer or guesses
when to switch. Both are rejected here. The supported performance contract is
instead explicit: static and localized work scale with actual changes;
layout containment bounds semantic layout coupling; repaint boundaries make
stable-content placement compositor-only; genuinely global paint/layout pays
for genuinely global work and remains reported rather than hidden.

The historical mixed global-placement result illustrates the same API boundary. A node
transform on one monolithic cached surface changes pixels, so every affected
mixed-family record moves and repaints. The declared repaint-boundary case
instead moves 100 cached chunks, changes zero child pixels, and measures 1.655
ms versus stock's 17.754 ms. Ordinary `UiTransform` retains its normal paint
semantics. No full-redraw fallback, area threshold, or promotion heuristic can
provide compositor-only placement without that explicit boundary contract.

Known portable test gaps remain. They are missing proofs, not known failures:

- containment still lacks differentials for min/max/aspect constraints and real
  glyph reflow. Flex-wrap reverse, right-to-left layout, percentages, absolute
  descendants, hidden nodes, intrinsic `ContentSize`, and scrollbar appearance
  and disappearance are covered;
- despawning a containment boundary itself and changing target cameras in a
  multi-camera contained hierarchy lack dedicated regression tests. Boundary
  reparenting, leaf reparenting, marker removal, target-scale changes, and
  ghost-node flattening are covered;
- the windowed matrix does not yet automate burst scenarios for text reflow,
  grid-track mutation, or repeated spawn/despawn, and timing thresholds remain
  target-product decisions;
- GPU acceptance deliberately compiles pipelines synchronously. Pending image,
  font-atlas, and material resources prove that owed damage survives delayed
  readiness, but real asynchronous pipeline compilation still needs a separate
  integration proof.

Some behavior cannot be established by portable automated tests:

- DRAM traffic, tile-memory behavior, energy use, and battery impact require
  profiling on each physical GPU architecture;
- swapchain presentation and the fused final blit must be exercised through a
  real window on every supported backend; automated pixel tests use image
  targets, although they do run Bevy's native pipelined render app;
- operating-system compositor behavior and partial presentation are hidden by
  `wgpu`;
- device/driver command-buffer failures may only be observable through
  backend-specific completion diagnostics;
- ordering an arbitrary third-party GPU writer before its retained UI consumer
  depends on that integration's render graph. The write-declaration mapping is
  unit-tested and Bevy camera writers have GPU differentials, but each custom
  writer must add its own graph-order/readback proof.

These are reported as measured platform results, never inferred from desktop
timings. Pixel correctness remains testable through image targets and device
readback even when the performance mechanism is opaque.

## Increment status

The external scope probes, serialized GPU harness, canonical paint records,
persistent atomic surface, exact mask repair, old/new coverage, two-dimensional
candidate index, sampled-image propagation, persistent prepared instances,
main-world dirty domains, layout containment, fused final composition, and the
runtime A/B stress matrix are implemented. The ordered retained compositor has
executable proofs for arbitrary sibling stacking, nesting, clips, post-flatten
opacity, hidden-subtree suspension, exact run memory accounting, sustained
animation, cached child paint, reflow without surviving-source raster, one
moving source among 64 static boundaries, coordinate-preserving run growth,
one masked submission per selected source across disjoint damage, target-local
stock material extraction, and immediate-mode material fallback and departure.
A boundary-free target allocates no run source; an all-boundary target allocates
no empty runs.

The remaining validation increment is a fresh optimized end-to-end sweep of the
ordered implementation followed by physical-device architecture and energy
sweeps. The 2026-08-12 timings above predate ordered sources. Desktop Vulkan
numbers cannot establish tile-memory traffic, Metal and mobile driver behavior,
or battery impact.

Cleanup is part of every increment: superseded paths, flags, thresholds, and
comments are removed before the next capability is added.
