# Incremental vascular growth: initial cost audit

Audit completed before implementation changes on `codex/vascular-incremental-growth`, base PR #310 at `4bcc361dbd19f770247a343c379285705f0f6738`. Remote master was fetched and inspected at `3660ddc27ffdf0c942bfe0038d2080feb45481dc`.

Notation: N is physical network entities; M is background cells; L is final numerical line cells; P is ranks; B is attempted growth batches; D is maximum ancestral depth; C is candidate background cells for one physical segment; I is crossed intervals; Q is returned spatial hits; A is the union of dependencies actually changed. Rejected proposals count towards B. An ordinary serial successful insertion adds two vessels and two nodes.

## Evidence and boundaries

The historical manifests and archived executable/library hashes were verified for strong-1000 and strong-10000. Both campaigns used clean source commit `76bb381187285e57bdc281a09f2bdd4c09a70703`; the diff from that commit to PR310 changes one line in growth_venous.cc and no growth-loop/commit/distribution/physiology/discretization source. Baseline measurements therefore describe the relevant implementation, with that explicit provenance qualification.

The cases use a unit cube, three uniform refinements (512 cells), seed23/24/25, seed point(.2,.5,.5), physical seed length.15, radius.003, viscosity.004, density exponent0, total tissue flow2e-9, port pressure12000, no calibration, coverage distance.2. Stored records have statuscomplete, returncode0, valid1 and exactly1000/10000 arterial terminals. BinarySHA256 d4a70de019707637c76897b44dd33723a5cc3a1805793abc784467096f8a06ca; librarySHA256559e03f47fe7a6e041f40bf6d134a133b8f2c2b3b5b28965dd70e0c29b9eb082. The manifest SHA256s are6a12f6e0e03af0abd2291ce7221dd1d480c0acc069742dd17199dfe59960cec4 andc8a1ac9f45d49c634d1a88f173572ee0aa95c9edb5e874e3dca72fd33dc02f79.

Serial generation medians:144.29592964s and14177.368884775s, ratio98.252036 for N ratio10. For seed23 at10000, redistribution/certification/physiology dominate as in the request. Phases nested inside commit must not be added to commit again. The archived manifests contain the individual measurements. This audit did not rerun the historical campaign.

## Repeated work inventory

| Item | Owning code and traversed state | Frequency and cost | Communication/state rebuilt | Incremental treatment and evidence |
|---|---|---|---|---|
| A: snapshot copies | growth_commit.cc copies NetworkData, mutable ids and root controls; repeated validate_data calls in commit, distribution, forest, paths and batch | Each attempted accepted-topology commit: Omega(N) copying and validation; cumulative Omega(sum N_b), hence quadratic in ordinary serial growth | Copies complete local graph including ghosts and complete control sets; validations revisit all local records | A provisional in-place journal records only inserted/erased/modified nodes, vessels, terminals and auxiliary metadata. Validate changed records plus explicit incidence/ownership dependencies; keep independent full final validation. Count copy attempts, visited records, journal sizes and rollback work. Proof is the explicit complete-container loops/copies, not a timing inference. |
| B: migration twice | growth_commit.cc migrate before and after evaluate_forest; distribution.cc authoritative_data, unique_owned_ids, source construction, point arrays, packet maps, halo and terminal ghosts | Two full passes per nonempty commit, plus startup: Omega(N); background cache/routing construction each time adds M-dependent setup | Global duplicate-id directory, authoritative node refresh, ParticleHandler relocation of every local node and every owned vessel midpoint, complete owner payloads and halo refresh; several sparse exchanges/allgathers/reductions | Keep persistent authority/incidence/ghost subscriptions and background routing. Relocate only new/moved nodes and changed midpoints through ParticleHandler. Radius-only changes update envelopes/subscribers without point relocation. Journal owned and ghost edits/version changes. Final redistribution remains an independent oracle. Count localized/migrated entities and ghost add/update/remove. Dominance measured historically; exact per-entity work will be instrumented. |
| C: full certification | growth_commit.cc constructs one BatchProposal per mutable vessel; growth_batch.cc validates snapshot, localizes every proposed endpoint, rebuilds obstacle tree, checks walls and conflicts | Every successful proposed commit: Omega(N) endpoint/wall work and index rebuilding, plus actual returned hits Q. Existing proposal-conflict RTree already avoids unconditional all-pairs | Complete endpoint particle search, coarse routing, whole proposal serialization and votes; boundary geometry also rebuilt | Certify only changed centerlines or increased envelopes/changed constraints. Reuse shrunk-envelope collision/wall certificates only with unchanged geometry/clearance policy; minimum radius still checked via maintained extrema. Root normalizations that increase untouched radii may require nonlocal work: count it rather than skip. Keep independent complete final certification. Count recertified vessels and exact pair/wall checks separately. |
| D: network indices | growth_loop.cc nearest packs tree from all mutable owned branches; candidate_obstacles packs all owned network, scans forest roots for max radius; producer packs returned obstacles again; batch packs all relevant obstacles | Every batch, including ranks without producer: Omega(N) traversal and bulk tree construction; query work adds logN+Q | Rank boxes and maxima rebuilt/reduced; candidate geometry serialized anew | Persistent native RTree<Segment,id>, native point/envelope indices and ordered radius bounds. Insert/remove/update only changed ids and ownership routes; centerline index unaffected by radius-only changes. Version remote entries and invalidate targeted subscribers. Producer temporary obstacle result indexing remains proportional to actual Q, not N, and may be reused by revision. Count full builds, inserted/removed entries, queries and hits. |
| E: background/boundary | BackgroundGeometry constructor loops all owned active cells and boundary faces and packs boundary patches; distribution and batch recreate GridTools::Cache/coarse covering trees | Per commit/per migration creates avoidable B*M work in fixed background; boundary work proportional to boundary face count | Coarse box allgathers, covering RTree allocation; immutable geometry reconstructed | One concrete persistent background context for a generate/calibration session; ctor remains noncollective, explicit collective initialization when routing is needed. Mapping/background lifetime and mesh-change invalidation are explicit. Share native deal.II caches and boundary tree; do not create a generic spatial framework. Count constructions/cell/face visits. |
| F: forest physiology | forest_state.cc rebuilds registry/incidence, rooted topology and leaf scale, upward/downward sweeps, normalized radii, flows/resistances and root metadata for every branch | Each commit: Omega(N) visits and materialization, with O(D) communication rounds. Unbalanced trees can have D=O(N), independent of N-balanced ordinary case | Native node-owner incidence packets, hash id registry, leaf-demand max reduction; upward and downward sparse packets/termination reductions; flows to terminal owners | Maintain parent/children/leaf/root associations, subtree demand and volume/resistance moments; update union of changed leaf/split ancestors. Keep lazy per-root radius normalization and per-system flow normalization. Materialize SI records once for public result/final solve/output. Old absolute Di updates may be nonlocal; API accepts explicit changed terminal ids with new absolute values and canonical total mass/revision. Count actual branch visits and roots whose increased normalization forces recertification. Preserve independent evaluate_forest oracle. |
| G: paths/optimizer depth | growth_paths.cc full validate_data, local cache recreated, ancestry fetched in global rounds; growth_candidate.cc recomputes moment sums and changed ancestors per optimizer trial; growth_loop cheap_geometry repeatedly searches paths for normals, ids and proposed obstacle demands | A path union is O(kD) records and O(D) rounds; cost includes avoidable O(N) validation. Nested changed-edge/path scans can be O(D^2) per trial, before actual geometric hits | Repeated remote ancestor records and root metadata, no retained revision cache | Persistent versioned ancestral records and direct id/endpoint lookup; retrieve missing/changed union only. Precompute per-candidate ancestor aggregates and node normals before optimization. No assumption of balanced trees; report A and depth separately. Count remote records, rounds and path comparisons. |
| H: venous choice | growth_loop.cc advances std::map iterator by index on each proposal; growth_venous.cc sends root/exclusion queries to every rank with available territory, nearest result count grows with excluded set, repeats after conflicts | Indexed map advance costs O(pending size), cumulative quadratic even without geometry; exclusions may create further superlinear work depending on rejections | Available-count allgather, O(P) query fanout per pending root, exclusion sets and nearest result serialization | Keep deterministic indexed/order-statistic territory selection with deletion and stable-id ordering (native/random-access facilities where adequate). Route nearest by persistent owner boxes, bounded exact nearest search and local admissibility cursor/version; do not approximate matches or drop failed candidates. Count inspected territories, exclusion records and recipients. Measurements needed to separate typical and adversarial exclusions. |
| I: segment intervals | discretization.cc traversed tests every candidate cell for every global cut interval; distributed_intervals rescans all reported intervals per combined cut; subdivide restarts from interval0 for each target; cell_resolution rescans all intervals; cuts inserted/erased in vector | Proven O(C^2), O(I^2), O(L*I), repeated O(I) resolution and O(L^2) shifts can occur for ONE long segment. Final discretization once; mesh-scale integration may make clipping part of repeated growth | Background segment routing and interval replies, plus local sort/search work | Reuse exact face crossing and native RTree tests; sorted event sweeps preserve finest-H/lowest-owner semantics, monotone prefix cursors for metric cuts, indexed range summaries for h min/max and ownership, output append/linked local edits rather than repeated vector shifts. Count predicates, events, interval visits and cut edits. Preserve all independent warped-Q1/tetra/interface/exterior oracles and add many-crossing counter oracles. |
| J: accessory state | full owner id registry, rank boxes, maxima, fresh identifiers, packet allocation/serialization; per-level readiness reductions in forest/paths | Mix of O(N) rebuilds, O(P) small control exchange, O(A) genuine dependencies and O(D) rounds. Serial follows same allocations and full loops | Identity directories, maxima, self and remote sparse exchanges, message buffers and copies | Persistent sharded id/owner directories and updated bounds, stable fresh-id allocator, retained buffers and explicit revisions. Distinguish source-visible logical records from actual MPI interposition. Existing benchmarks/vascular_mpi_profile.cc already records calls and bytes via PMPI without extra reductions; retain it for transport measurements. Add cheap operation counters in existing per-rank growth buckets, no new per-candidate reductions. |
| K: one-time work | final discretize, tissue coverage, reference Poiseuille solve, archives/tables/native transfer/full validation; calibration retries all generation | Outside ordinary insertion loop. O(N+M+L) geometry/algebra assembly plus solver/query costs; may still have independent bad algorithms but cannot explain repeated commit cost | Final distributed algebra and final output/metadata communication | Time and count separately; retain independent complete correctness gates and actual public SI materialization. Share immutable background among calibration attempts but reset mutable execution/territory state per attempt. Do not report nested phase times as additive or call all time MPI. |

## First correctness gate and integration contract

Use the existing small distributed forest split/commit tests, plus a new direct comparison of incremental updates against complete evaluate_forest/redistribute/full batch certification at each accepted step. Include rejection after each stage and verify graph, physiology, owner/ghost sets, native RTree queries and revisions return to the pre-attempt state. Full public generation and the actual standalone application remain the vertical consumers; no alternate approximate generator.

The parent mesh-adaptation worker owns Length scale, exact H profiles/clearance semantics and Voronoi integration. This branch owns persistent execution, demand-delta propagation, journaling and certification invalidation. Agreed draft API: absolute new Di by existing terminal id, new Di in GrowthSplit; one preview per tissue territory, joint chosen-centre demand update, canonical global total demand and quadrature/snapshot revision; no separate competing propagation algorithm. Geometry and demand revisions must participate in the same commit or rollback. Root-factor increases are explicitly nonlocal when unavoidable.

No implementation is complete at this checkpoint. Historical timings establish priority, not proof that any proposed representation meets the target. Do not launch the 1000/2000/5000/10000 campaign until structural counters show that ordinary insertions no longer rebuild the full graph/background.

## Operation counters

Each per-rank `growth-<rank>.csv` keeps its existing columns and appends the
`GrowthOperation` counters. Counts include unsuccessful attempts and require
no MPI calls. Sum rank-local counts when comparing distributed work. Timings
inside commit overlap commit time; they must not be added again.

`discretization-work-<rank>.csv` records final line-discretization operations
separately. A cache construction counts explicit native cache objects in instrumented
ImmersX code; ParticleHandler-internal caches are outside that counter.
cell and face visits count the inspected local cells and owned-cell faces.
Index hits count returned entries before application filtering. Copy attempts
count attempted physical map insertions, including duplicate endpoint inserts.
Localizations count submitted particles, while migrations count owned records
whose owner changes. Ghost updates include received refreshes and additions.
Physiology visits count registration, upward sweep, downward sweep and final
materialization. Path visits count inspected ancestral records. Interval cut
moves count vector elements shifted by insertion or erasure.

Zero removal/journal counters describe the current full-rebuild implementation;
they do not establish that incremental transactions exist. MPI calls and bytes
are measured independently by the existing benchmark PMPI interposer. Full
validation, final reference solve and output remain independent correctness
checks. The counters are diagnostics and do not change acceptance decisions.

## Measured small-case structural baseline

The original synthetic configuration above, with seeds 23, 24 and 25, gives
identical counts for each seed:

| Terminal count | Submitted vessel midpoints | Fully certified vessels |
|---:|---:|---:|
| 32 | 2,048 | 1,024 |
| 64 | 8,192 | 4,096 |
| 128 | 32,768 | 16,384 |

There are no failed joint commits in these cases. Initial normalization and
each accepted insertion relocate every midpoint twice and certify every mutable
vessel once: the counts are exactly `2*N*N` and `N*N`. Doubling the target
quadruples both counts. These measurements describe the implementation before
incremental updates; they establish a structural baseline rather than a speedup.
The benchmark script retains work per rank and sums per stage and per run,
with final discretization counts in a separate field. It can still validate
historical archives that predate these columns.

## Persistent centerline indices

The growth loop now builds native deal.II RTrees once for owned physical
centerlines and for each system's mutable candidates. A committed batch
publishes changed and released owned vessel identifiers. Native migration
notifications carry these identifiers to the new owner; no comparison of
complete old and new snapshots is required. Each affected index entry is
removed or inserted in logarithmic work. Rank bounds come from the native
RTree bounds without visiting its leaves. Radius changes leave these indices
valid because their keys contain only centerlines.

A fixed global bound over the real root controls and frozen input radii bounds
all subsequent mutable Murray radii. Candidate obstacle routing uses this
bound, avoiding a repeated full scan of radii. All exact envelope and collision
checks still use current physical radii. The background and mapping remain
fixed for the context lifetime; changing either requires a new context.

The loop also reuses its boundary geometry for commit certification. The
independent full commit can still construct its own context when called alone.
Rejected commits publish no index changes. Tests compare stored entries,
intersection queries and nearest results with exhaustive searches before and
after a distributed split and after a rejected transaction.

At this intermediate milestone, migration and certification still rebuild
complete state. Producer indices over actual query hits are also temporary.
The persistent indices remove repeated candidate/owned-obstacle bulk builds;
they do not yet remove all quadratic growth work or all background cache
constructions.

The installed deal.II ParticleHandler implementation also constructs its own
GridTools cache and allocates a cell-indexed vector at construction. Its
`clear()` rebuilds that vector, and `update_cached_numbers()` may scan all
active cells when particle cell order needs sorting. Reusing an external cache
alone therefore does not remove the repeated background work. The next
migration milestone must retain native search state and avoid these paths;
cache-construction counters alone are insufficient evidence while fresh
ParticleHandlers remain in the insertion loop.

Nearest-candidate queries use the native incremental RTree query iterator,
stopping after the requested cutoff and its equal-distance results. Sorting
these results by distance and physical identifier makes candidate ties
independent of the tree's insertion history. A conservative floating-point
guard only broadens the queried results; the final ordering uses the physical
distance without changing it. Work includes the returned cutoff ties and the
first result beyond the cutoff. No bounding-box expansion scans distant rank
partitions to find ties. The exact historical graph can change where the old
bulk nearest query chose an arbitrary subset of equal-distance vessels;
benchmark comparisons must retain attempt and geometric-work counts.
