"""
    splitdataframe(df::DataFrame, p::Real)

Partitionne en un ensemble d'entraînement et un ensemble de validation un DataFrame.

### Arguments
- `df::DataFrame` : Un DataFrame
- `p::Real` : La proportion (entre 0 et 1) de données dans l'ensemble d'entraînement.

### Détails

La fonction renvoie deux DataFrames, un pour l'ensemble d'entraînement et l'autre pour l'ensemble de validation.

### Exemple

\```
 julia> splitdataframe(df, .7)
\```

"""
function splitdataframe(df::DataFrame, p::Real)
   @assert 0 <= p <= 1

    n = size(df,1)

    ind = shuffle(1:n)

    threshold = Int64(round(n*p))

    indTrain = sort(ind[1:threshold])

    indTest = setdiff(1:n,indTrain)

    dfTrain = data[indTrain,:]
    dfTest = data[indTest,:]

    return dfTrain, dfTest

end
