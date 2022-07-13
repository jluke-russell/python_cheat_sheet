# Filtering a dataset 
luke_history = names.query("name == 'Luke' & year >= 1965 & year <= 1999 ")[['name','year','MD']]
#print(luke_history)

# Making a Table for Markdown
print(luke
    .head(5)
    .filter(["name", "year", "MD"])
    .to_markdown(index=False))

# Multiple names (or states) and it's graph
bible_names = names.query('name in ["Mary", "Martha", "Paul", "Peter"]  & year >= 1920  & year <= 2000')
#print(bible_names)

bible_chart = (alt.Chart(bible_names).properties(title='Popularity of Select Christian Names')
  .mark_line()
    .encode(
        x= alt.X('year', axis= alt.Axis(format="d", title="Year")),
        y= alt.Y('Total', axis = alt.Axis(title='Total')),
        color = 'name'
  )
)

# adds vertical and horizontal line 
xrule = (
    alt.Chart()
    .mark_rule(color="cyan", strokeWidth=2)
    .encode(x=alt.datum(alt.DateTime(year=2006, month="November")))
)

yrule = (
    alt.Chart().mark_rule(strokeDash=[12, 6], size=2).encode(y=alt.datum(350))
)

# puts it all together 
bible_chart + xrule + yrule