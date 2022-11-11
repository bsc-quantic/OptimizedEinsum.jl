abstract type Pattern end

function parse(out, a)
    error("not implemented yet")

end

function parse(out, a, b)
    error("not implemented yet")
end

parse(::Type{<:Pattern}, args...) = nothing

struct InnerProduct <: Pattern end

parse(::Type{InnerProduct}, out, a, b) = (x = setdiff(a ∩ b, out); isempty(x) ? nothing : x)

struct OuterProduct <: Pattern end

parse(::Type{OuterProduct}, out, a, b) = nothing

struct HadamardProduct <: Pattern end

parse(::Type{HadamardProduct}, out, a, b) = (x = ∩(out, a, b); isempty(x) ? nothing : x)

struct Trace <: Pattern end

parse(::Type{Trace}, out, a) = (x = setdiff(nonunique(a), out); isempty(x) ? nothing : x)

struct Permutation <: Pattern end

parse(::Type{Permutation}, out, a) = begin
    # other type of operations not allowed
    @assert issetequal(a, out)

    return indexin(a, out)
end

struct Reduction <: Pattern end

# TODO not implemented yet
parse(::Type{Reduction}, out, a) = nothing

struct MultiPattern <: Pattern
    patterns::Vector{Pattern}
end

# TODO not implemented yet
parse(::Type{MultiPattern}, out, a) = nothing