import json
import uuid
from neo4j import GraphDatabase
from typing import List, Dict, Tuple

def generate_uuid():
    return str(uuid.uuid4())

def escape_for_cypher(text):
    """Escape text for Cypher queries"""
    if text is None:
        return ""
    
    # Replace backslashes first (important order)
    text = text.replace('\\', '\\\\')
    # Replace quotes
    text = text.replace('"', '\\"')
    # Replace newlines
    text = text.replace('\n', '\\n')
    # Replace tabs
    text = text.replace('\t', '\\t')
    # Replace carriage returns
    text = text.replace('\r', '\\r')
    
    return text

def generate_cypher_from_json(json_file_path: str, batch_size: int = 100) -> Tuple[List[str], Dict]:
    """Generate Cypher queries with batch processing"""
    
    # Read JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file).get("chunks", [])
    
    print(f"Processing {len(data)} chunks...")
    
    # Process in batches to avoid memory issues
    all_queries = []
    
    # Phase 1: Create all nodes using UNWIND for batch insert
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        node_data = []
        
        for chunk in batch:
            project_id = chunk.get('project_id', '')
            file_path = chunk.get('file_path', '')
            class_name = chunk.get('class_name', '')
            method_name = chunk.get('method_name', '')
            chunk_type = chunk.get('chunk_type', '')
            content = chunk.get('content', '')
            endpoints = chunk.get('endpoints', [])
            summary = chunk.get('summary', '')
            endpoint = str(endpoints)
            
            # Determine node type
            if chunk_type == "controller" and method_name and endpoint:
                node_type = "EndpointNode"
            elif method_name:
                node_type = "MethodNode"
            else:
                node_type = "ClassNode"
            
            node_data.append({
                'node_type': node_type,
                'project_id': project_id,
                'file_path': file_path,
                'class_name': class_name,
                'method_name': method_name,
                'chunk_type': chunk_type,
                'content': content,
                'endpoint': endpoint,
                'summary': summary
            })
        
        # Create batch insert query
        batch_query = """
        UNWIND $nodes AS node
        CALL apoc.create.node([node.node_type], {
            project_id: node.project_id,
            file_path: node.file_path,
            class_name: node.class_name,
            method_name: node.method_name,
            chunk_type: node.chunk_type,
            content: node.content,
            endpoint: node.endpoint,
            summary: node.summary
        }) YIELD node AS created_node
        RETURN count(created_node)
        """
        
        all_queries.append((batch_query, {'nodes': node_data}))
    
    # Phase 2: Create relationships using batch processing
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        # Collect all relationships for this batch
        call_rels = []
        called_by_rels = []
        implement_rels = []
        implemented_by_rels = []
        extend_rels = []
        extended_by_rels = []
        use_rels = []
        used_by_rels = []
        
        for chunk in batch:
            class_name = chunk.get('class_name', '')
            method_name = chunk.get('method_name', '')
            
            # CALL relationships
            for call in chunk.get('calls', []):
                if '.' in call:
                    called_class, called_method = call.rsplit('.', 1)
                    call_rels.append({
                        'source_class': class_name,
                        'source_method': method_name,
                        'target_class': called_class,
                        'target_method': called_method
                    })
            
            # CALLED_BY relationships
            for called in chunk.get('called_by', []):
                if '.' in called:
                    caller_class, caller_method = called.rsplit('.', 1)
                    called_by_rels.append({
                        'source_class': class_name,
                        'source_method': method_name,
                        'target_class': caller_class,
                        'target_method': caller_method
                    })
            
            # IMPLEMENT relationships
            for impl in chunk.get('implements', []):
                 if '.' in impl:
                    impl_class, impl_method = impl.rsplit('.', 1)
                    implement_rels.append({
                        'source_class': class_name,
                        'source_method': method_name,
                        'target_class': impl_class,
                        'target_method': impl_method
                    })
            
            # IMPLEMENTED_BY relationships
            for impl_by in chunk.get('implemented_by', []):
                implemented_by_rels.append({
                    'source_class': class_name,
                    'source_method': method_name,
                    'target_class': impl_by,
                    'target_method': method_name
                })
            
            # EXTEND relationships
            for extend in chunk.get('extends', []):
                if '.' in extend:
                    extend_class, extend_method = extend.rsplit('.', 1)
                    extend_rels.append({
                        'source_class': class_name,
                        'source_method': method_name,
                        'target_class': extend_class,
                        'target_method': extend_method
                    })
            
            # EXTENDED_BY relationships
            for extended_by in chunk.get('extended_by', []):
                extended_by_rels.append({
                    'source_class': class_name,
                    'source_method': method_name,
                    'target_class': extended_by,
                    'target_method': method_name
                })
            
            # USE relationships
            for var in chunk.get('vars', []):
                use_rels.append({
                    'source_class': class_name,
                    'source_method': method_name,
                    'target_class': var
                })

            for used_by in chunk.get('vars', []):
                used_by_rels.append({
                    'source_class': used_by,
                    'target_class': class_name,
                    'target_method': method_name
                })

        # Create batch queries for each relationship type
        if call_rels:
            call_query = """
            UNWIND $relationships AS rel
            MATCH (source {class_name: rel.source_class, method_name: rel.source_method})
            MATCH (target {class_name: rel.target_class, method_name: rel.target_method})
            MERGE (source)-[:CALL]->(target)
            """
            all_queries.append((call_query, {'relationships': call_rels}))
        
        if called_by_rels:
            called_by_query = """
            UNWIND $relationships AS rel
            MATCH (source {class_name: rel.source_class, method_name: rel.source_method})
            MATCH (target {class_name: rel.target_class, method_name: rel.target_method})
            MERGE (source)-[:CALLED_BY]->(target)
            """
            all_queries.append((called_by_query, {'relationships': called_by_rels}))
        
        if implement_rels:
            implement_query = """
            UNWIND $relationships AS rel
            MATCH (source {class_name: rel.source_class, method_name: rel.source_method})
            MATCH (target {class_name: rel.target_class, method_name: rel.target_method})
            MERGE (source)-[:IMPLEMENT]->(target)
            """
            all_queries.append((implement_query, {'relationships': implement_rels}))
        
        if implemented_by_rels:
            implemented_by_query = """
            UNWIND $relationships AS rel
            MATCH (source {class_name: rel.source_class, method_name: rel.source_method})
            MATCH (target {class_name: rel.target_class, method_name: rel.target_method})
            MERGE (source)-[:IMPLEMENTED_BY]->(target)
            """
            all_queries.append((implemented_by_query, {'relationships': implemented_by_rels}))
        
        if extend_rels:
            extend_query = """
            UNWIND $relationships AS rel
            MATCH (source {class_name: rel.source_class, method_name: rel.source_method})
            MATCH (target {class_name: rel.target_class, method_name: rel.target_method})
            MERGE (source)-[:EXTEND]->(target)
            """
            all_queries.append((extend_query, {'relationships': extend_rels}))
        
        if extended_by_rels:
            extended_by_query = """
            UNWIND $relationships AS rel
            MATCH (source {class_name: rel.source_class, method_name: rel.source_method})
            MATCH (target {class_name: rel.target_class, method_name: rel.target_method})
            MERGE (source)-[:EXTENDED_BY]->(target)
            """
            all_queries.append((extended_by_query, {'relationships': extended_by_rels}))

        if use_rels:
            use_query = """
            UNWIND $relationships AS rel
            MATCH (source {class_name: rel.source_class})
            WHERE (rel.source_method IS NULL OR source.method_name = rel.source_method)
            MATCH (target {class_name: rel.target_class})
            WHERE target.method_name IS NULL
            MERGE (source)-[:USE]->(target)
            """
            all_queries.append((use_query, {'relationships': use_rels}))

        if used_by_rels:
            used_by_query = """
            UNWIND $relationships AS rel
            MATCH (source {class_name: rel.source_class})
            MATCH (target {class_name: rel.target_class})
            WHERE target.method_name = rel.target_method
            MERGE (source)-[:USED_BY]->(target)
            """
            all_queries.append((used_by_query, {'relationships': used_by_rels}))

    return all_queries

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def execute_queries_batch(self, queries_with_params: List[Tuple[str, Dict]], max_retries: int = 3):
        """Execute queries with better error handling and memory management"""
        
        with self.driver.session() as session:
            for i, (query, params) in enumerate(queries_with_params):
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        result = session.run(query, params)
                        # Consume result to free memory
                        result.consume()
                        
                        print(f"Executed batch {i+1}/{len(queries_with_params)}")
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        print(f"Error executing batch {i+1} (attempt {retry_count}/{max_retries}): {str(e)}")
                        
                        if retry_count >= max_retries:
                            print(f"Failed to execute batch {i+1} after {max_retries} attempts")
                            raise e
                        
                        # Wait before retry
                        import time
                        time.sleep(1)
    
    def create_indexes(self):
        """Create indexes for better performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:EndpointNode) ON (n.class_name, n.method_name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:MethodNode) ON (n.class_name, n.method_name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:ClassNode) ON (n.class_name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:EndpointNode) ON (n.class_name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:MethodNode) ON (n.class_name)",
        ]
        
        with self.driver.session() as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                    print(f"Created index: {index_query}")
                except Exception as e:
                    print(f"Error creating index: {str(e)}")

def main():
    # Neo4j connection details
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "your_password"
    
    # JSON file path
    json_file_path = "my_output.json"
    
    # Configuration
    BATCH_SIZE = 50  # Reduce batch size if still getting OOM
    
    print("Starting Neo4j import process...")
    
    # Connect to Neo4j
    neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Create indexes first for better performance
        print("Creating indexes...")
        neo4j_conn.create_indexes()
        
        # Generate optimized queries
        print("Generating optimized queries...")
        queries_with_params = generate_cypher_from_json(json_file_path, BATCH_SIZE)
        
        print(f"Generated {len(queries_with_params)} batch queries")
        
        # Execute queries in batches
        print("Executing queries...")
        neo4j_conn.execute_queries_batch(queries_with_params)
        
        print("Import completed successfully!")
        
    except Exception as e:
        print(f"Error during import: {str(e)}")
        raise e
    finally:
        neo4j_conn.close()

# Example usage
if __name__ == "__main__":
    main()