# julia fista_golden_compare.jl <baseline file> <new file>
using Serialization
same(a, b) = a isa AbstractArray ? (size(a) == size(b) && all(isequal.(a, b))) :
             a isa NamedTuple ? (keys(a) == keys(b) && all(same(a[k], b[k]) for k in keys(a))) : isequal(a, b)
old, new = deserialize(ARGS[1]), deserialize(ARGS[2])
bad = String[]
for name in sort(collect(union(keys(old), keys(new))))
    if !haskey(old, name) || !haskey(new, name)
        push!(bad, "$name: missing in one recording"); continue
    end
    a, b = old[name], new[name]
    for f in sort(collect(union(keys(a), keys(b))))
        (haskey(a, f) && haskey(b, f) && same(a[f], b[f])) || push!(bad, "$name: field $f differs")
    end
end
foreach(println, first(bad, 40))
println(isempty(bad) ? "IDENTICAL: all $(length(old)) cases match bit for bit" : "DIFFERENT: $(length(bad)) mismatches")
exit(isempty(bad) ? 0 : 1)
