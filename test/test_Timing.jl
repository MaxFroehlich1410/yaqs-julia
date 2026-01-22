using Test
using Yaqs

@testset "Timing" begin
    @testset "Disabled mode is no-op" begin
        Yaqs.reset_timing!()
        Yaqs.enable_timing!(false)
        ts = Yaqs.begin_scope!()
        @test ts === nothing
        # end_scope! should accept nothing
        @test Yaqs.end_scope!(ts) === nothing
    end

    @testset "Enabled mode records counts" begin
        Yaqs.reset_timing!()
        Yaqs.set_timing_print_each_call!(false)
        Yaqs.enable_timing!(true)

        ts = Yaqs.begin_scope!()
        @test ts !== nothing
        @test Yaqs.Timing._active_stats() === ts

        # Record a key at least once
        x = Yaqs.@t :unit_timing_key sum(abs2, randn(256))
        @test x ≥ 0

        Yaqs.end_scope!(ts; header="unit")
        @test Yaqs.Timing._active_stats() === nothing

        # Global stats should contain the key with count ≥ 1
        g = Yaqs.Timing._GLOBAL_TIMING
        @test haskey(g.counts, :unit_timing_key)
        @test g.counts[:unit_timing_key] ≥ 1

        # print_timing_summary! should run
        io = IOBuffer()
        redirect_stdout(io) do
            Yaqs.print_timing_summary!(header="timing")
        end
        out = String(take!(io))
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

