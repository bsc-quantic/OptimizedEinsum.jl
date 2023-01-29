@testset "ContractionPath" begin
    using OptimizedEinsum: ContractionPath, signatures, labels, rank

    output = Symbol[] # TODO fix #23 and try `[:e]`
    inputs = [[:h, :c, :f], [:a], [:d, :b], [:g, :d, :b, :a, :f], [:e, :h, :c, :g]]
    size_dict = Dict(:a => 5, :b => 5, :f => 8, :d => 2, :e => 2, :c => 4, :h => 9, :g => 8)
    ssapath = [(4, 3), (5, 1), (7, 6), (8, 2)]
    path = ContractionPath(ssapath, inputs, output, size_dict)

    n = length(inputs)
    @test length(path) == 4
    @test size.((path,), 1:n) == map(i -> prod(getindex.((size_dict,), i)), inputs)
    @test size(path, n + 1) == 320
    @test size(path, n + 2) == 128
    @test size(path, n + 3) == 10
    @test size(path, n + 4) == 2

    @test signatures(path) == [inputs..., [:g, :a, :f], [:e, :g, :f], [:e, :a], [:e]]
    @test all(x -> ((i, sign) = x; labels(path, i) == sign), enumerate(signatures(path)))

    @test maximum(path) do (i, j)
        rank(labels(path, i), labels(path, j), size_dict, output)
    end == 3

    @test sum(path) do (i, j)
        flops(labels(path, i), labels(path, j), size_dict, output)
    end == 3200 + 4608 + 640 + 10
end