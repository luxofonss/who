MATCH (endpoint:EndpointNode {class_name: "PronunciationController", method_name: "getAccuracy"})

CALL apoc.path.expandConfig(endpoint, {
  relationshipFilter: "CALL>|IMPLEMENTED_BY>|EXTENDED_BY>|USE>",
  minLevel: 1,
  maxLevel: 20,
  bfs: true,
  uniqueness: "NODE_GLOBAL",
  filterStartNode: false
})
YIELD path

WITH endpoint, path, nodes(path) AS node_list, relationships(path) AS rel_list
WHERE NONE(i IN range(0, size(rel_list)-2) WHERE type(rel_list[i]) = 'USE' AND size(rel_list) > i + 1)

WITH endpoint, path, node_list, rel_list,
     [i IN range(0, size(rel_list)-1) |
        CASE
          WHEN type(rel_list[i]) = 'CALL'
               AND node_list[i+1].method_name IS NOT NULL
          THEN node_list[i+1]
          WHEN type(rel_list[i]) IN ['IMPLEMENTED_BY', 'EXTENDED_BY']
               AND node_list[i+1].method_name IS NOT NULL
               AND node_list[i].method_name IS NOT NULL
               AND toString(node_list[i+1].method_name) = toString(node_list[i].method_name)
          THEN node_list[i+1]
          WHEN type(rel_list[i]) = 'USE'
               AND node_list[i+1].method_name IS NULL
          THEN node_list[i+1]
          ELSE null
        END
     ] AS filtered_nodes

RETURN endpoint, path, 
       [node IN filtered_nodes WHERE node IS NOT NULL] AS visited_nodes_per_path
ORDER BY path