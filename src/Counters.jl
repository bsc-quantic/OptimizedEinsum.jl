using Base.Iterators: flatten

flops(a, b, size) = flops(a, b, size, ())
flops(a, b, size, keep) = prod(ind -> size[ind], flatten((a ∪ b, ∩(keep, a, b))), init=one(BigInt))

rank(a, b, size, keep) = length(symdiff(a, b) ∪ ∩(keep, a, b))

removedsize(a, b, size, keep) = prod(size[ind] for ind ∈ symdiff(a, b) ∪ (keep ∩ a) ∪ (keep ∩ b); init=1) - prod(size[ind] for ind ∈ a) - prod(size[ind] for ind ∈ b)