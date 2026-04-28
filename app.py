from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FuncFormatter

app = Flask(__name__)

# Data Awal (Hardcoded agar aman saat dideploy)
data_awal = {
    'provinsi': ['ACEH', 'SUMUT', 'JABAR', 'JATENG', 'JATIM', 'DKI', 'SULSEL', 'PAPUA'],
    'min_upah': [3166460, 2522609, 1841487, 1812935, 1891567, 4641854, 3165876, 3561932],
    'gk': [579201, 548290, 448113, 423274, 443425, 738955, 385311, 624311]
}
df = pd.DataFrame(data_awal)

@app.route('/', methods=['GET', 'POST'])
def index():
    global df
    if request.method == 'POST':
        # Ambil input dari user
        prov = request.form.get('provinsi').upper()
        upah = float(request.form.get('upah'))
        gk = float(request.form.get('gk'))
        # Tambah data baru ke dataframe
        new_row = {'provinsi': prov, 'min_upah': upah, 'gk': gk}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # --- LOGIKA CLUSTERING ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['min_upah', 'gk']])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # --- BIKIN GRAFIK (VERSI DI-UPGRADE) ---
    plt.figure(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    
    # Looping buat gambar tiap cluster
    for i, c in enumerate(sorted(df['cluster'].unique())):
        cluster_data = df[df['cluster'] == c]
        plt.scatter(cluster_data['min_upah'], cluster_data['gk'], 
                    label=f'Cluster {c}', color=colors[i], 
                    s=150, edgecolor='black', alpha=0.8)

    # Penjelasan Visual
    plt.title('Visualisasi Clustering K-Means\nKesejahteraan Provinsi', fontsize=14, fontweight='bold')
    plt.xlabel('Upah Minimum (Rp)', fontsize=12)
    plt.ylabel('Garis Kemiskinan (Rp)', fontsize=12)
    
    # Format angka biar gak pakai 1e6 (Scientific)
    plt.ticklabel_format(style='plain', axis='both') 
    
    # Fungsi pembantu buat label "Jt" atau "Rb"
    def jutaan_formatter(x, pos):
        return f'{x/1000000:.1f} Jt' if x >= 1000000 else f'{int(x/1000)} Rb'
    
    plt.gca().xaxis.set_major_formatter(FuncFormatter(jutaan_formatter))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(jutaan_formatter))

    plt.grid(True, linestyle='--', alpha=0.5) 
    plt.legend(title='Kelompok', loc='upper left')
    
    # Simpan grafik ke memori untuk dikirim ke HTML
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('result.html', data=df.to_dict(orient='records'), plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
