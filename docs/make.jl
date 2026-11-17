using Documenter
using DynamicalResilience

# Source and edit links need a git remote; outside a git checkout, build without them.
ingit = success(pipeline(`git -C $(@__DIR__) rev-parse`; stdout = devnull, stderr = devnull))

makedocs(;
         modules = [DynamicalResilience],
         authors = "Simon Frost",
         sitename = "DynamicalResilience.jl",
         checkdocs = :exports,
         remotes = ingit ? Dict() : nothing,
         format = Documenter.HTML(;
                                  edit_link = ingit ? "main" : nothing,
                                  prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://ecorecipes.github.io/DynamicalResilience.jl"),
         pages = ["Home" => "index.md", "API Reference" => "api.md"])

if get(ENV, "CI", "false") == "true"
    deploydocs(; repo = "github.com/ecorecipes/DynamicalResilience.jl.git", push_preview = true)
end
