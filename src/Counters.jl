using Base.Iterators: flatten
using Distributions: Normal, truncated
using Random: rand

flops(a, b, size) = prod(ind -> size[ind], a ∪ b, init=1)
flops(a, b, size, keep) = prod(ind -> size[ind], flatten((a ∪ b, ∩(keep, a, b))), init=1)

rank(a, b, size, keep) = length(symdiff(a, b) ∪ ∩(keep, a, b))

removedsize(a, b, size, keep) = prod(size[ind] for ind ∈ symdiff(a, b) ∪ (keep ∩ a) ∪ (keep ∩ b); init=1) - prod(size[ind] for ind ∈ a) - prod(size[ind] for ind ∈ b)

removedsize_noise(a, b, size, keep) = rand(truncated(Normal(1.0, 0.01),0,10),1)[1] * (prod(size[ind] for ind ∈ symdiff(a, b) ∪ (keep ∩ a) ∪ (keep ∩ b); init=1) - prod(size[ind] for ind ∈ a) - prod(size[ind] for ind ∈ b))