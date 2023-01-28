@testset "Utils" begin
    @testset "pathtype" begin
        using OptimizedEinsum: pathtype

        @test pathtype([(1, 2)]) == :ssa
        @test pathtype([(1, 2), (3, 4)]) == :ssa
        @test pathtype([(2, 4), (1, 5), (3, 6)]) == :ssa

        @test pathtype([(1, 2), (1, 3)]) == :linear
        @test pathtype([(1, 2), (2, 3)]) == :linear
    end

    @testset "ssa_to_linear" begin
        using OptimizedEinsum: ssa_to_linear

        @test ssa_to_linear([(1, 4), (3, 5), (2, 6)]) == [[1, 4], [2, 3], [1, 2]]
        @test ssa_to_linear([[1, 4], [3, 5], [2, 6]]) == [[1, 4], [2, 3], [1, 2]]
    end

    @testset "linear_to_ssa" begin
        using OptimizedEinsum: linear_to_ssa

        @test linear_to_ssa([(1, 4), (2, 3), (1, 2)]) == [[1, 4], [3, 5], [2, 6]]
        @test linear_to_ssa([[1, 4], [2, 3], [1, 2]]) == [[1, 4], [3, 5], [2, 6]]
    end

    # TODO test `ssa_path_cost`

    @testset "nonunique" begin
        using OptimizedEinsum: nonunique

        @test isempty(nonunique(1, 2, 3))
        @test issetequal(nonunique(1, 2, 3, 1), (1,))
        @test isempty(nonunique([1, 2, 3]))
        @test issetequal(nonunique([1, 2, 3, 1]), (1,))
        @test isempty(nonunique((1, 2, 3)))
        @test issetequal(nonunique((1, 2, 3, 1)), (1,))

        @test isempty(nonunique(:a, :b, :c))
        @test issetequal(nonunique(:a, :b, :c, :a), (:a,))
        @test isempty(nonunique([:a, :b, :c]))
        @test issetequal(nonunique([:a, :b, :c, :a]), (:a,))
        @test isempty(nonunique((:a, :b, :c)))
        @test issetequal(nonunique((:a, :b, :c, :a)), (:a,))

        @test isempty(nonunique([[:a, :b, :c], [:a, :c], [:c]]))
        @test issetequal(nonunique([[:a, :b, :c], [:a, :c], [:c], [:a, :c]]), ([:a, :c],))

        @test isempty(nonunique([(:a, :b, :c), (:a, :c), (:c,)]))
        @test issetequal(nonunique([(:a, :b, :c), (:a, :c), (:c,), (:a, :c)]), ((:a, :c),))

        @test isempty(nonunique([Set([:a, :b, :c]), Set([:a, :c]), Set([:c])]))
        @test issetequal(nonunique([Set([:a, :b, :c]), Set([:a, :c]), Set([:c]), Set([:a, :c])]), (Set([:a, :c]),))
    end

    @testset "popvalue!" begin
        using OptimizedEinsum: popvalue!

        d = Dict(:a => 1, :b => 2, :c => 3)

        @test popvalue!(d, 1) == :a
        @test :a ∉ keys(d)
        @test popvalue!(d, 1) == nothing

        # TODO test if repeated value
    end
end