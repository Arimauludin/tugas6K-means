from flask import Flask, render_template
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def index():
    # Data manual singkat agar PASTI JALAN
    data = {
        'provinsi': ['ACEH', 'SUMUT', 'JABAR', 'JATENG', 'JATIM', 'DKI', 'SULSEL', 'PAPUA'],
        'tahun': [2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022],
        'min_upah': [3166460, 2522609, 1841487, 1812935, 1891567, 4641854, 3165876, 3561932],
        'gk': [579201, 548290, 448113, 423274, 443425, 738955, 385311, 624311]
    }
    df = pd.DataFrame(data)
    
    # K-Means
    X = df[['min_upah', 'gk']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    return render_template('result.html', data=df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, port=8080)