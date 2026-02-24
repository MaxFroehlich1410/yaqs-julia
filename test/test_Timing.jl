# Unit tests for the `Yaqs.Timing` instrumentation utilities.
#
# These tests validate that timing scopes are no-ops when disabled, that enabled scopes record
# timing keys and call counts, and that summary printing functions execute without error.
#
# Args:
#     None
#
# Returns:
#     Nothing: Defines `@testset`s covering timing enable/disable, scoping, and internal helpers.
using Test
using Yaqs

@testset "Timing" begin
    @testset "Disabled mode is no-op" begin
        Yaqs.Timing.reset_timing!()
        Yaqs.Timing.enable_timing!(false)
        ts = Yaqs.Timing.begin_scope!()
        @test ts === nothing
        # end_scope! should accept nothing
        @test Yaqs.Timing.end_scope!(ts) === nothing
    end

    @testset "Enabled mode records counts" begin
        Yaqs.Timing.reset_timing!()
        Yaqs.Timing.set_timing_print_each_call!(false)
        Yaqs.Timing.enable_timing!(true)

        ts = Yaqs.Timing.begin_scope!()
        @test ts !== nothing
        @test Yaqs.Timing._active_stats() === ts

        # Record a key at least once
        x = Yaqs.Timing.@t :unit_timing_key sum(abs2, randn(256))
        @test x ≥ 0

        Yaqs.Timing.end_scope!(ts; header="unit")
        @test Yaqs.Timing._active_stats() === nothing

        # Global stats should contain the key with count ≥ 1
        g = Yaqs.Timing._GLOBAL_TIMING
        @test haskey(g.counts, :unit_timing_key)
        @test g.counts[:unit_timing_key] ≥ 1

        # print_timing_summary! should run
        out = mktemp() do path, io
            redirect_stdout(io) do
                Yaqs.Timing.print_timing_summary!(header="timing")
            end
            close(io)
            return read(path, String)
        end
        @test occursin("timing", out)
    end

    @testset "Internal helpers" begin
        ts1 = Yaqs.Timing.TimingStats()
        ts2 = Yaqs.Timing.TimingStats()
        Yaqs.Timing._timing_add!(ts1, :a, UInt64(10))
        Yaqs.Timing._timing_add!(ts1, :a, UInt64(5))
        Yaqs.Timing._timing_add!(ts2, :a, UInt64(7))
        Yaqs.Timing._timing_add!(ts2, :b, UInt64(3))

        Yaqs.Timing._timing_merge!(ts1, ts2)
        @test ts1.times_ns[:a] == UInt64(22)
        @test ts1.counts[:a] == 3
        @test ts1.times_ns[:b] == UInt64(3)
        @test ts1.counts[:b] == 1
    end
end

