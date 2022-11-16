using Base.Iterators: flatten

flops(a, b, size) = prod(size[ind] for ind ∈ (a ∪ b))
flops(a, b, size, keep) = prod(size[ind] for ind ∈ flatten((a ∪ b, ∩(keep, a, b))))

rank(a, b, size, keep) = length(symdiff(a, b) ∪ ∩(keep, a, b))

removedsize(a, b, size, keep) = prod(size[ind] for ind ∈ symdiff(a, b) ∪ (keep ∩ a) ∪ (keep ∩ b); init=1) - prod(size[ind] for ind ∈ a) - prod(size[ind] for ind ∈ b)