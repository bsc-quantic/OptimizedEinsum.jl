@testset "Visualization" begin
    using CairoMakie
    using OptimizedEinsum: rand_equation, contractpath, Greedy
    using NetworkLayout: Spring
    using Makie: FigureAxisPlot, AxisPlot

    @testset "plot`" begin
        output, inputs, size_dict = rand_equation(10, 2)
        path = contractpath(Greedy, inputs, output, size_dict)

        @test plot(path) isa FigureAxisPlot
        @test plot(path; labels=true) isa FigureAxisPlot
        @test plot(path; layout=Spring(dim=3)) isa FigureAxisPlot

        @test begin
            f = Figure()
            plot!(f[1,1], path) isa AxisPlot
        end

        @test begin
            f = Figure()
            plot!(f[1,1], path; labels=true) isa AxisPlot
        end

        @test begin
            f = Figure()
            plot!(f[1,1], path; layout=Spring(dim=3)) isa AxisPlot
        end
    end
end