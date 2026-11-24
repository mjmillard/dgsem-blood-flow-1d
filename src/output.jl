# Figure style and tables. Each table is declared once as a TableSpec and rendered
# as aligned ASCII (stdout) and as a Markdown pipe table (file).

const FIG_DPI = 150
const GUIDEFONT = 14
const TICKFONT = 11
const LEGENDFONT = 11

"Shared Plots defaults. Figures carry no titles; captions are in the thesis."
function thesis_plot_style!()
    Plots.default(dpi = FIG_DPI, guidefontsize = GUIDEFONT, tickfontsize = TICKFONT,
                  legendfontsize = LEGENDFONT)
end

# Critical PNG chunks; tRNS is kept because dropping it would change the pixels.
const PNG_KEEP_CHUNKS = ("IHDR", "PLTE", "tRNS", "IDAT", "IEND")

# Drop ancillary chunks (bKGD, tIME, ...) so the file depends on the pixels only
# and pdfTeX can embed the image data verbatim. Chunks are copied byte for byte.
function strip_png_ancillary!(path::AbstractString)
    bytes = read(path)
    out = IOBuffer()
    write(out, @view bytes[1:8])
    i = 9
    while i <= length(bytes)
        len = Int(ntoh(only(reinterpret(UInt32, bytes[i:(i + 3)]))))
        stop = i + 11 + len
        String(bytes[(i + 4):(i + 7)]) in PNG_KEEP_CHUNKS && write(out, @view bytes[i:stop])
        i = stop + 1
    end
    write(path, take!(out))
    return path
end

function thesis_savefig(plt, path::AbstractString)
    savefig(plt, path)
    endswith(lowercase(path), ".png") && strip_png_ancillary!(path)
    return path
end

"""
    TableColumn(key, header; fmt = :sci, digits = 3)

`fmt` is one of `:sci` (`digits` significant digits), `:fixed`, `:pct` (fraction
shown as a percent), `:pctsigned`, `:int` or `:str` (`digits` decimal places for
the fixed-point formats).
"""
struct TableColumn
    key::Symbol
    header::String
    fmt::Symbol
    digits::Int
end

TableColumn(key::Symbol, header::AbstractString; fmt::Symbol = :sci, digits::Int = 3) =
    TableColumn(key, String(header), fmt, digits)

"""
    TableSpec(title, columns, rows; notes = String[])

`rows` are NamedTuples looked up by column key; a missing key, `nothing` or `NaN`
renders as a blank cell.
"""
struct TableSpec
    title::String
    columns::Vector{TableColumn}
    rows::Vector{<:NamedTuple}
    notes::Vector{String}
end

TableSpec(title, columns, rows; notes = String[]) =
    TableSpec(String(title), collect(TableColumn, columns), collect(NamedTuple, rows),
              collect(String, notes))

is_blank(v) = v === nothing || (v isa Real && isnan(v))

function fmt_cell(v, col::TableColumn; blank::String = "-")
    is_blank(v) && return blank
    col.fmt === :str && return string(v)
    col.fmt === :int && return @sprintf("%d", v)
    col.fmt === :sci && return @sprintf("%.*e", max(col.digits - 1, 0), v)
    col.fmt === :fixed && return @sprintf("%.*f", col.digits, v)
    col.fmt === :pct && return @sprintf("%.*f%%", col.digits, 100 * v)
    col.fmt === :pctsigned && return @sprintf("%+.*f%%", col.digits, 100 * v)
    error("unknown column format :$(col.fmt)")
end

md_escape(s::AbstractString) = replace(s, "|" => "\\|")

pad(s, w, leftalign) = leftalign ? rpad(s, w) : lpad(s, w)

function print_ascii(spec::TableSpec; io::IO = stdout)
    cells = [[fmt_cell(get(row, col.key, nothing), col) for row in spec.rows]
             for col in spec.columns]
    widths = [maximum(length, vcat(spec.columns[j].header, cells[j]))
              for j in eachindex(spec.columns)]
    leftaligned = [col.fmt === :str for col in spec.columns]
    line(vals) = " " * join((pad(vals[j], widths[j], leftaligned[j])
                             for j in eachindex(spec.columns)), "  ")

    rule = "="^96
    println(io, "\n", rule)
    println(io, " ", spec.title)
    println(io, rule)
    println(io, line([col.header for col in spec.columns]))
    println(io, "-"^96)
    for i in eachindex(spec.rows)
        println(io, line([cells[j][i] for j in eachindex(spec.columns)]))
    end
    println(io, rule)
    for note in spec.notes
        println(io, " ", note)
    end
    isempty(spec.notes) || println(io, rule)
    return nothing
end

function print_markdown(io::IO, spec::TableSpec)
    cells = [[md_escape(fmt_cell(get(row, col.key, nothing), col; blank = "—"))
              for row in spec.rows] for col in spec.columns]
    widths = [max(3, maximum(length, vcat(md_escape(spec.columns[j].header), cells[j])))
              for j in eachindex(spec.columns)]
    leftaligned = [col.fmt === :str for col in spec.columns]
    row_line(vals) = "| " * join((pad(vals[j], widths[j], leftaligned[j])
                                  for j in eachindex(spec.columns)), " | ") * " |"

    println(io, "### ", spec.title)
    println(io)
    println(io, row_line([md_escape(col.header) for col in spec.columns]))
    println(io, "|", join((leftaligned[j] ? ":" * "-"^(widths[j] + 1) :
                           "-"^(widths[j] + 1) * ":"
                           for j in eachindex(spec.columns)), "|"), "|")
    for i in eachindex(spec.rows)
        println(io, row_line([cells[j][i] for j in eachindex(spec.columns)]))
    end
    if !isempty(spec.notes)
        println(io)
        for note in spec.notes
            println(io, "- ", note)
        end
    end
    println(io)
    return nothing
end

"""
    emit_tables(specs, dir, name; title = "", print = true)

Print the tables to stdout (unless `print = false`) and write them all to
`dir/name.md`.
"""
function emit_tables(specs::AbstractVector{TableSpec}, dir::AbstractString,
                     name::AbstractString; title::AbstractString = "", print::Bool = true)
    print && foreach(print_ascii, specs)
    path = joinpath(dir, name * ".md")
    open(path, "w") do io
        isempty(title) || (println(io, "# ", title); println(io))
        println(io, "<!-- Generated by dgsem-blood-flow-1d -- do not edit by hand. -->")
        println(io)
        foreach(spec -> print_markdown(io, spec), specs)
    end
    @info "Wrote $(basename(path)) to $dir"
    return path
end
