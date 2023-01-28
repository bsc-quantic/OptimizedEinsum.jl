@testset "Counters" begin
    size_dict = Dict(:o => 6, :b => 9, :n => 6, :j => 6, :d => 6, :e => 6, :k => 4, :c => 4, :h => 3, :g => 5, :l => 9, :m => 7, :f => 5, :a => 2, :i => 2)

    @testset "rank" begin
        using OptimizedEinsum: rank

        @test rank([:o, :j, :e], [:o], size_dict, []) == 2
        @test rank([:n, :c, :k, :a], [:n, :h, :a], size_dict, []) == 3
        @test rank([:b, :d, :h], [:o, :a, :k], size_dict, []) == 6

        @test rank([:o, :j, :e], [:o], size_dict, [:o]) == 3
        @test rank([:n, :c, :k, :a], [:n, :h, :a], size_dict, (:h, :n, :k)) == 4
        @test rank([:b, :d, :h], [:o, :a, :k], size_dict, Set([:d, :a])) == 6

        @test rank((:o, :j, :e), (:o,), size_dict, []) == 2
        @test rank((:n, :c, :k, :a), [:n, :h, :a], size_dict, ()) == 3
        @test rank((:b, :d, :h), (:o, :a, :k), size_dict, Set{Symbol}()) == 6

        @test rank((:o, :j, :e), (:o,), size_dict, [:o]) == 3
        @test rank((:n, :c, :k, :a), [:n, :h, :a], size_dict, (:h, :n, :k)) == 4
        @test rank((:b, :d, :h), (:o, :a, :k), size_dict, Set([:d, :a])) == 6


        @test rank(Set([:o, :j, :e]), Set([:o]), size_dict, []) == 2
        @test rank(Set([:n, :c, :k, :a]), Set([:n, :h, :a]), size_dict, ()) == 3
        @test rank(Set([:b, :d, :h]), Set([:o, :a, :k]), size_dict, Set{Symbol}()) == 6

        @test rank(Set([:o, :j, :e]), Set([:o]), size_dict, [:o]) == 3
        @test rank(Set([:n, :c, :k, :a]), Set([:n, :h, :a]), size_dict, (:h, :n, :k)) == 4
        @test rank(Set([:b, :d, :h]), Set([:o, :a, :k]), size_dict, Set([:d, :a])) == 6
    end

    @testset "flops" begin
        using OptimizedEinsum: flops

        @test flops([:o, :j, :e], [:o], size_dict, []) == 216
        @test flops([:n, :c, :k, :a], [:n, :h, :a], size_dict, []) == 576
        @test flops([:b, :d, :h], [:o, :a, :k], size_dict, []) == 7776

        @test flops([:o, :j, :e], [:o], size_dict, [:o]) == 1296
        @test flops([:n, :c, :k, :a], [:n, :h, :a], size_dict, (:h, :n, :k)) == 3456
        @test flops([:b, :d, :h], [:o, :a, :k], size_dict, Set([:d, :a])) == 7776

        @test flops((:o, :j, :e), (:o,), size_dict, []) == 216
        @test flops((:n, :c, :k, :a), [:n, :h, :a], size_dict, ()) == 576
        @test flops((:b, :d, :h), (:o, :a, :k), size_dict, Set{Symbol}()) == 7776

        @test flops((:o, :j, :e), (:o,), size_dict, [:o]) == 1296
        @test flops((:n, :c, :k, :a), [:n, :h, :a], size_dict, (:h, :n, :k)) == 3456
        @test flops((:b, :d, :h), (:o, :a, :k), size_dict, Set([:d, :a])) == 7776


        @test flops(Set([:o, :j, :e]), Set([:o]), size_dict, []) == 216
        @test flops(Set([:n, :c, :k, :a]), Set([:n, :h, :a]), size_dict, ()) == 576
        @test flops(Set([:b, :d, :h]), Set([:o, :a, :k]), size_dict, Set{Symbol}()) == 7776

        @test flops(Set([:o, :j, :e]), Set([:o]), size_dict, [:o]) == 1296
        @test flops(Set([:n, :c, :k, :a]), Set([:n, :h, :a]), size_dict, (:h, :n, :k)) == 3456
        @test flops(Set([:b, :d, :h]), Set([:o, :a, :k]), size_dict, Set([:d, :a])) == 7776
    end

    @testset "removedsize" begin
        using OptimizedEinsum: removedsize

        @test removedsize([:o, :j, :e], [:o], size_dict, []) == -186
        @test removedsize([:n, :c, :k, :a], [:n, :h, :a], size_dict, []) == -180
        @test removedsize([:b, :d, :h], [:o, :a, :k], size_dict, []) == 7566

        @test removedsize([:o, :j, :e], [:o], size_dict, [:o]) == -6
        @test removedsize([:n, :c, :k, :a], [:n, :h, :a], size_dict, (:h, :n, :k)) == 60
        @test removedsize([:b, :d, :h], [:o, :a, :k], size_dict, Set([:d, :a])) == 7566

        @test removedsize((:o, :j, :e), (:o,), size_dict, []) == -186
        @test removedsize((:n, :c, :k, :a), [:n, :h, :a], size_dict, ()) == -180
        @test removedsize((:b, :d, :h), (:o, :a, :k), size_dict, Set{Symbol}()) == 7566

        @test removedsize((:o, :j, :e), (:o,), size_dict, [:o]) == -6
        @test removedsize((:n, :c, :k, :a), [:n, :h, :a], size_dict, (:h, :n, :k)) == 60
        @test removedsize((:b, :d, :h), (:o, :a, :k), size_dict, Set([:d, :a])) == 7566


        @test removedsize(Set([:o, :j, :e]), Set([:o]), size_dict, []) == -186
        @test removedsize(Set([:n, :c, :k, :a]), Set([:n, :h, :a]), size_dict, ()) == -180
        @test removedsize(Set([:b, :d, :h]), Set([:o, :a, :k]), size_dict, Set{Symbol}()) == 7566

        @test removedsize(Set([:o, :j, :e]), Set([:o]), size_dict, [:o]) == -6
        @test removedsize(Set([:n, :c, :k, :a]), Set([:n, :h, :a]), size_dict, (:h, :n, :k)) == 60
        @test removedsize(Set([:b, :d, :h]), Set([:o, :a, :k]), size_dict, Set([:d, :a])) == 7566
    end
end