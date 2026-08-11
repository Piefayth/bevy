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
- a static UI performs no main-world UI work, no extraction, and no raster work;
  if the game renders a fresh world frame, only the irreducible composition of
  visible cached UI remains;
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

Known likely expansion points are:

1. `bevy_ui`: stock layout, stack construction, and clipping walk static trees.
   [Issue #22909](https://github.com/bevyengine/bevy/issues/22909) describes the
   same idle-cost and root-invalidation problem. We will first test whether an
   external plugin can gate the public `UiSystems` sets and handle placement
   separately.
2. `bevy_core_pipeline`: a fresh world frame still needs cached UI composited.
   We will first test a final-writer interposition that preserves Bevy's output
   attachment and presentation bookkeeping, avoiding a core-pipeline fork.

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
The focused `bevy_ui` patch now splits the original function into public,
ordered `ui_layout_system` and `ui_geometry_system` systems. The former owns
Taffy synchronization and computation; the latter owns placement, scrolling,
rounding, outlines, radii, and derived render geometry. This is the second
justified `bevy_ui` expansion: it lets a `UiTransform` or scroll animation skip
Taffy without duplicating Bevy internals. Static-tree quiescence alone would not
have justified it.

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

The core-pipeline probe found a public and simpler interposition point than
changing `CameraOutputMode` after target preparation. Bevy schedules the final
`upscaling` writer as an ordinary system in `Core2d`/`Core3d`, and the public
schedule API can remove a system by its function type. A third-party plugin can
therefore remove that one system and install its own final writer after
`Core2dSystems::PostProcess`/`Core3dSystems::PostProcess`. The proof in
`bevy_ui_render_retained/tests/core_pipeline_scope.rs` removes exactly one stock
writer and exactly one replacement.

This preserves Bevy's `ViewTarget`, output attachment, and later presentation
bookkeeping without vendoring `bevy_core_pipeline`. The scheduling proof does
not establish that the replacement blit has correct pixels or that every
window backend presents it; image-target readback will establish the former,
while the latter still requires a window integration test on each supported
backend.

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
`bevy_ecs` or `bevy_core_pipeline` fork is required.

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

Box shadows use a `BoxShadowInfrastructurePlugin` and a shared
`resolve_box_shadow` function for the same reason. The retained crate owns
candidate extraction and canonical records while the stock and retained paths
share target-relative value resolution, queueing, preparation, and the shader.
This is still entirely inside the focused `bevy_ui_render` replacement patch.

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
| Measure | `Node`, content size, text metrics, target scale | run layout for the affected dependency root |
| Placement | `UiTransform`, scroll offset | update spatial properties and affected descendant coverage |
| Paint | colors, glyphs, borders, image selection | rebuild only affected paint records and repair their pixels |
| Composite | retained-island transform or opacity | update compositor properties, no island repaint |
| Resource | image/atlas/material/sampled-surface revision | repaint exact readers and their filter reach |

For existing Bevy components, the intended semantics are:

- `Node` changes layout;
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

For each damaged region:

1. Preflight every pipeline, bind group, texture, and buffer needed by the
   repair.
2. Keep old retained state and damage owed if preflight fails.
3. Wipe the damaged region with blending disabled.
4. Replay every intersecting record bottom-up, with the repair scissor active.
5. Commit the new retained state only after commands for the complete repair
   were encoded.

The region representation is a canonical union of integer physical rectangles.
No subpixel tolerance is required. A record's conservative coverage includes
antialiasing, shadows, outlines, filters, and texture sampling reach. Old
coverage remains authoritative until the new pixels commit, so subpixel drift
cannot accumulate invisibly.

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

Composition need not sample a fullscreen transparent layer. The renderer can
retain occupied coverage tiles or regions and issue a static instanced draw for
only nonempty UI coverage in the same render pass as Bevy's final world blit:

```text
draw world/upscaled image
draw cached occupied UI regions
```

This avoids an additional attachment load/store and makes sampling cost
proportional to visible UI coverage. A fullscreen translucent UI remains
fullscreen work by definition.

The public `wgpu::Surface` presentation API has no damage-region parameter, so
portable platform partial-present behavior cannot currently be promised.

- [`wgpu::Surface`](https://docs.rs/wgpu/latest/wgpu/struct.Surface.html)

A general subtree repaint boundary cannot be implemented by cutting a hole in
one cached parent texture and compositing the subtree afterward. A later sibling
may paint above the boundary; flattening all non-boundary paint into one texture
loses the ordering point at which the boundary layer must be inserted. The exact
representation is a compositor display list alternating cached paint chunks
with boundary surfaces. Nested boundaries recurse, and each boundary defines an
atomic stacking context. This is now a prerequisite for the boundary increment;
a component that merely suppresses transform invalidation is rejected because
it preserves stale pixels, while a topmost-only restriction is too narrow for
the intended API.

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
  text transforms, sampled surfaces, resize, and resource unavailability;
- a deterministic adversarial scene rendered once with retention and once with
  full redraw, followed by an exact pixel comparison;
- counters asserted by tests: entities extracted, paint records changed, items
  and prepared quads replayed, damaged pixels, surfaces repaired, and composite
  coverage.

Every regression test must first fail with the named defect reintroduced.

Performance is tested continuously, not inferred from the correctness suite.
Bevy's repository-wide benchmark crate uses Criterion and named baselines; the
retained renderer adds a `ui` benchmark target there and follows the existing
`cargo bench --bench <name> -- --save-baseline <baseline>` workflow. There was
no dedicated Bevy UI benchmark target when this work began, although the
benchmark crate already depended on `bevy_ui`.

Every incremental capability has three benchmark states over identical UI
trees:

- quiet: no UI input changes;
- localized: one leaf changes in exactly one dirty domain;
- full: every relevant leaf changes.

Wall-clock samples catch constant-factor regressions. Deterministic counters
catch complexity regressions even when timing noise is high. Benchmarked work
counts include roots and entities visited, entities extracted, paint records
compared/changed, bytes uploaded, records and prepared quads replayed, damaged
physical pixels, surface repairs, composite instances, and composite pixel
coverage. The quiet case is required to report zero for every counter except
unavoidable composition when another renderer produces a fresh target frame.

- [Bevy benchmark instructions](../benches/README.md)

The initial short-run baseline on 2026-08-10 used the stock `UiPlugin` over a
flat tree. At 10,000 nodes this machine measured approximately 1.18 ms quiet,
4.52 ms for one changed leaf, and 5.77 ms when every leaf changed. These are
local comparison points, not portable claims. More importantly, the quiet
100/1,000/10,000 results scale from roughly 0.08/0.17/1.18 ms, confirming the
static stock path remains linear in tree size. The benchmark source is
`benches/benches/bevy_ui/layout.rs`.

The main-world quiescence benchmark runs the same quiet/localized/full trees
through `RetainedUiMainWorldPlugin`. On this machine, a short 10,000-node quiet
run fell from approximately 1.145 ms to 0.200 ms after layout, geometry, stack,
and clipping were gated. A localized width change still requires Bevy's
whole-root Taffy and geometry walk. After splitting placement from Taffy, a
localized `UiTransform` change at 10,000 nodes fell from approximately 1.27 ms
on the stock path to 0.60 ms on the retained path. It still walks the whole UI
tree for geometry, so it remains linear and is not the desired endpoint. These
are local comparison points, not portable claims. The quiet remainder includes
the exact `Changed<T>` scans and other ungated `PostUpdate` systems; it is not
described as zero CPU work. Atomic counters separately prove zero Taffy,
geometry, stack, and clipping walks on static and paint-only frames. Tests also
prove width changes wake Taffy, geometry, and clipping; `UiTransform` wakes
geometry and clipping but not Taffy; `ZIndex` wakes only stack; `OverrideClip`
wakes only clipping; hierarchy changes wake every recursive domain; removal
cleans the stack; and custom systems in the public UI sets continue to run
normally.

The first retained-core benchmark on the same machine measured quiet repair
planning at roughly 6 ns and one canonical record change plus exact damage at
roughly 0.26 microseconds, invariant across 100/1,000/10,000 retained records.
A full 10,000-record change took roughly 1.73 ms. This benchmark covers only the
canonical record map and exact damage journal; it does not yet include Bevy
extraction, GPU upload, raster repair, or composition.

After exact multi-region coverage was added, Criterion caught an avoidable
full-change regression to roughly 1.97 ms: `upsert` cloned both coverage sets.
Recording through disjoint borrows removed those copies. Replacing a four-rect
inline array—which enlarged every hash-map record—with compact
`Empty | One | Many` storage then measured roughly 2 ns quiet, 0.27
microseconds for one localized change, and 1.63 ms for 10,000 full changes in a
short Criterion run. The common zero/one-region cases allocate nothing.

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
beneath it. The named GPU image proof changes a 10-by-10 image, repairs exactly
100 pixels, and replays exactly those two intersecting records bottom-up.

Records use `Changed<T>` candidate nomination, bit-exact canonical values,
stable render entities, and old-union-new damage. Each exact, non-overlapping
damage rectangle is wiped to transparent and rebuilt from only the sorted
phase items whose physical bounds intersect it. A localized background proof
moves a 10-by-10 leaf across a larger background: exactly 200 pixels are
repaired and only the three necessary item-region intersections are replayed.

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

Solid borders and outlines retain four fixed edge records per visible family.
This deliberately separates damage identity from GPU draw grouping. A first
attempt drew every edge independently; GPU comparison rejected it because
equal-color corner ties then alpha-blended more than once. The accepted path
keeps edge identity stable, but merges equal canonical commands at replay time
by OR-ing their border flags. Changing the left edge out of an equal-color
group therefore damages only that edge's 140-pixel rounded-corner reach, while
the other three edges still replay as one command under the damage scissor.
Equal-color borders, distinct-color regrouping, outlines, and component
removal are byte-identical to stock in named GPU tests.

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
for storage only when they use it. Damage, item intersection, and cached
composite coverage all consume the same region set. A unit proof keeps the gap
between two regions absent from damage.

The owned UI layer is double-buffered. A repair encodes into the inactive
texture and replaces the visible texture only after every region succeeds.
Each slot carries a committed generation; bringing an older slot current
copies only damage committed since that generation, rather than copying the
whole layer. Failed or not-yet-compiled work remains owed. On a quiet
background-only frame, GPU counters prove zero additional repairs, repair
pixels, and replayed items while composition continues. An equal component
replacement nominates and compares exactly one record but changes zero records
and causes zero repairs; a real color change repairs exactly that record's old
and new physical coverage.

Each layer is sized in viewport-local physical pixels. UI repair uses that
local surface directly; only composition applies the camera viewport. A
nonzero-origin viewport is byte-identical to stock, and moving an unchanged
viewport is proven to cause zero paint candidates and zero repair work. This
is placement-only motion of a retained surface.

Composition caches the exact union of possibly nontransparent item bounds when
a repair commits. Quiet frames reuse those regions and issue one scissored
texture draw per non-overlapping region; they do not rebuild the union. A lone
10-by-10 UI composites 100 pixels rather than the 4,096-pixel test viewport.
Two disjoint translucent 10-by-10 regions remain byte-identical to stock,
preserve the untouched gap, and composite exactly 200 pixels in two draws. An
empty committed layer stops compositing entirely, and a camera with no visible
UI allocates no layer textures. If a layer is lost or no longer matches its
viewport size or target format while its records remain unchanged, the whole
viewport is added to owed damage and those records replay on the following
extraction. Replacing the scissored draws with one static instanced region draw
remains a constant-factor optimization; it does not change pixel coverage or
invalidation semantics.

Some behavior cannot be established by portable automated tests:

- DRAM traffic, tile-memory behavior, energy use, and battery impact require
  profiling on each physical GPU architecture;
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

## Increment order

1. Prove external `UiSystems` quiescence and final-writer interposition. These
   decide crate scope.
2. Build a serialized GPU-test harness and retained-vs-full differential scene.
3. Retain canonical paint records and process only changed/removed entities.
4. Add a persistent window UI surface, full repair on change, and complete skip
   on a quiet frame.
5. Add exact damage regions, wipe/replay, old/new coverage, and per-record
   culling.
6. Add transform/clip/effect/scroll property separation and declared retained
   boundaries.
7. Propagate committed damage from retained offscreen surfaces to their exact
   image readers.
8. Gate or replace main-world layout, stack, and clipping work by dirty domain.
9. Perform runtime-selectable on-device architecture sweeps and retain only
   mechanisms justified by measurements.

Cleanup is part of every increment: superseded paths, flags, thresholds, and
comments are removed before the next capability is added.
