using Base.Iterators: flatten

flops(a, b, size) = prod(size[ind] for ind ∈ (a ∪ b))
flops(a, b, size, keep) = prod(size[ind] for ind ∈ flatten((a ∪ b, ∩(keep, a, b))))