import numpy as np
import json
from sklearn.cluster import DBSCAN
import database

# Thresholds
CLUSTERING_TOLERANCE = 0.5  # DBSCAN eps - how similar faces in ONE video must be to cluster
MATCHING_THRESHOLD = 0.5    # How similar a cluster must be to DB profile to match

class Profiler:
    def __init__(self):
        pass

    def run(self, video_id, detections):
        if not detections:
            return

        embeddings = [d['embedding'] for d in detections]
        
        # 1. Cluster detections within THIS video
        # Persons might appear multiple times. We group them.
        clusters = self._cluster_internal(embeddings)
        
        # 2. Load existing profiles from DB
        db_persons = database.get_all_persons() # list of (id, name, json_emb)
        
        # 3. Match Clusters to Profiles
        for cluster_id, detection_indices in clusters.items():
            if cluster_id == -1:
                # OPTIONAL: Handle noise/unknowns. For now, we ignore "noise" or treat as separate?
                # Usually DBSCAN -1 means "didn't cluster". We can treat them as singletons.
                for idx in detection_indices:
                    self._resolve_identity(video_id, [detections[idx]], db_persons)
            else:
                cluster_dets = [detections[i] for i in detection_indices]
                self._resolve_identity(video_id, cluster_dets, db_persons)

    def _cluster_internal(self, embeddings):
        """
        Returns dictionary: { cluster_label: [index1, index2...], ... }
        """
        if len(embeddings) == 0:
            return {}
            
        # DBSCAN metric='euclidean' on 128d embeddings
        clt = DBSCAN(eps=CLUSTERING_TOLERANCE, min_samples=3, metric="euclidean")
        labels = clt.fit_predict(embeddings)
        
        results = {}
        for idx, label in enumerate(labels):
            if label not in results:
                results[label] = []
            results[label].append(idx)
            
        # If min_samples is high, many might be -1.
        # Fallback: if we only have 1-2 frames of someone, they end up as -1.
        # We should treat -1s as individual checks.
        
        return results

    def _resolve_identity(self, video_id, cluster_detections, db_persons):
        """
        Decides if this cluster is Person A (from DB) or a new Person B.
        """
        # Calculate centroid of this cluster
        embs = np.array([d['embedding'] for d in cluster_detections])
        centroid = np.mean(embs, axis=0)

        best_match_id = None
        best_dist = 100.0 # Infinity

        for person in db_persons:
            pid, name, p_emb_json = person
            p_emb = np.array(json.loads(p_emb_json))
            
            # Euclidean distance
            dist = np.linalg.norm(centroid - p_emb)
            
            if dist < MATCHING_THRESHOLD and dist < best_dist:
                best_dist = dist
                best_match_id = pid
        
        final_person_id = best_match_id
        
        if final_person_id is None:
            # Create NEW person
            print(f"  -> Detected NEW Identity. Creating Profile...")
            final_person_id = database.create_person(centroid)
            
            # Need to refresh db_persons list if we want to match subsequent clusters to this new guy?
            # For simplicity, we assume one person usually doesn't split into 2 distinct clusters in one video that don't match each other.
            # But if they do, we ideally want to link them. 
            # Current logic: The NEXT cluster will check DB, see this new guy, and match him! 
            # So we append this new guy to local db_persons copy.
            db_persons.append((final_person_id, "New", json.dumps(centroid.tolist())))
        else:
            print(f"  -> Matched Person ID {final_person_id} (Dist: {best_dist:.3f})")

        # Log appearances
        # We verify just 1 "best" image for the person? Or log ALL frames? 
        # Logging ALL frames can be huge. Let's log just the center-most (lowest blur?) or every detection.
        # Requirement: "track... 1000s of incidents". Usually means we want every occurrence.
        
        for d in cluster_detections:
            database.log_appearance(video_id, final_person_id, d['timestamp'], d['crop_path'])
