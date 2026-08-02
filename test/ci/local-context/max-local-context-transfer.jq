# BuildKit emits vertex names and transfer status updates in separate events.
# Join them by vertex digest and return the largest local://context transfer.
(reduce .[] as $event (
  {names: {}, transfers: {}};
  reduce ($event.vertexes // [])[] as $vertex (
    .;
    .names[$vertex.digest] = $vertex.name
  )
  | reduce ($event.statuses // [])[] as $status (
      .;
      if $status.name == "transferring" then
        .transfers[$status.vertex] = ([.transfers[$status.vertex] // 0, $status.current] | max)
      else
        .
      end
    )
)) as $state
| [
    $state.transfers
    | to_entries[]
    | select($state.names[.key] == "local://context")
    | .value
  ] as $transfers
| if ($transfers | length) == 0 then
    error("BuildKit did not report a local://context transfer")
  else
    $transfers | max
  end
