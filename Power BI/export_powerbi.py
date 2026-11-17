import json
import pandas as pd
import requests

print("🔄 Export des données vers Power BI...")
print("="*50)

# 1. Récupérer les communautés
print("📥 Récupération des communautés...")
response = requests.get("http://127.0.0.1:8000/graph/communities")
data = response.json()

communities_data = []
for comm_id, comm_info in data['communities'].items():
    communities_data.append({
        'community_id': comm_id,
        'size': comm_info['size'],
        'members': ', '.join(comm_info['members']) if isinstance(comm_info['members'], list) else comm_info['members'],
        'topic': comm_info.get('topic', 'Unknown')
    })
df_communities = pd.DataFrame(communities_data)
df_communities.to_csv('communities.csv', index=False, encoding='utf-8-sig')
print(f"   ✅ {len(df_communities)} communautés exportées")

# 2. Récupérer les nœuds du graphe
print("📥 Récupération des nœuds...")
response = requests.get("http://127.0.0.1:8000/graph/aura")
data = response.json()

nodes_data = []
for node in data.get('nodes_sample', []):
    nodes_data.append({
        'node_name': node['name'],  # Garde node_name
        'color': node.get('color', '#1e3a5f')
    })
df_nodes = pd.DataFrame(nodes_data)
# Force les noms de colonnes
df_nodes.columns = ['node_name', 'color']
df_nodes.to_csv('nodes.csv', index=False, encoding='utf-8-sig')

df_nodes = pd.DataFrame(nodes_data)
df_nodes.to_csv('nodes.csv', index=False, encoding='utf-8-sig')
print(f"   ✅ {len(df_nodes)} nœuds exportés")

# 3. Récupérer les relations
print("📥 Récupération des relations...")
relationships_data = []
for rel in data.get('relationships_sample', []):
    relationships_data.append({
        'source': rel['source'],
        'relation_type': rel['rel_type'],
        'target': rel['target']
    })
df_rels = pd.DataFrame(relationships_data)
df_rels.to_csv('relationships.csv', index=False, encoding='utf-8-sig')
print(f"   ✅ {len(df_rels)} relations exportées")

# 4. Récupérer les points PCA (optionnel mais cool)
print("📥 Récupération des points PCA...")
try:
    response = requests.get("http://127.0.0.1:8000/embeddings/pca")
    data = response.json()
    
    pca_data = []
    for point in data.get('points', []):
        pca_data.append({
            'x': point['x'],
            'y': point['y'],
            'cluster': point['cluster'],
            'chunk_preview': point['chunk_preview'][:50]
        })
    df_pca = pd.DataFrame(pca_data)
    df_pca.to_csv('pca_points.csv', index=False, encoding='utf-8-sig')
    print(f"   ✅ {len(df_pca)} points PCA exportés")
except Exception as e:
    print(f"   ⚠️ PCA non disponible: {e}")

print("="*50)
print("🎯 EXPORT TERMINÉ !")
print("\nFichiers créés :")
print("   📄 communities.csv")
print("   📄 nodes.csv")
print("   📄 relationships.csv")
print("   📄 pca_points.csv")
print("\n👉 Ouvre ces fichiers dans Power BI maintenant !")