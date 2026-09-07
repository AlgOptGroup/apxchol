using SHA, TOML
let root=dirname(dirname(pathof(Laplacians))), source=TOML.parsefile(joinpath(@__DIR__,"public-source.toml"))
    for (name, expected) in source["files"]
        bytes2hex(open(SHA.sha256,joinpath(root,name))) == expected || error("Public source mismatch: $name")
    end
    println("VERIFIED_PUBLIC_SOURCE ",root)
end
