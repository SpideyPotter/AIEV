# Server Instructions (GTA Inference)

Use these steps to start the cloud segmentation server and connect from GTA side.

## 1) Go to repo

```bash
cd /home/dgx-s-bmu-cse-230480/final/FinalTry
```

## 2) Start/refresh inference server on Kubernetes

```bash
kubectl apply -f k8s/job_inference.yaml
kubectl apply -f k8s/service_inference.yaml
```

If an old Job is stuck, recreate it:

```bash
kubectl delete job idd-panoptic-inference --ignore-not-found
kubectl apply -f k8s/job_inference.yaml
```

## 3) Check server pod is running

```bash
kubectl get pods -l app=idd-panoptic-inference -o wide
kubectl logs -f job/idd-panoptic-inference
```

Expected: pod should be `Running` and logs should show `Uvicorn running on http://0.0.0.0:8000`.

## 4) Start port-forward on server machine

Run this and keep terminal open:

```bash
kubectl port-forward --address 0.0.0.0 svc/idd-panoptic-inference 8000:8000
```

## 5) Health check (same machine)

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"ok":true}
```

## 6) Test one image request

```bash
curl -X POST "http://127.0.0.1:8000/segment" -F "image=@road.jpg" -o outputs/test_response.json
```

## 7) If GTA PC cannot access `10.1.0.176:8000`, use SSH tunnel from GTA PC

On GTA PC:

```bash
ssh -L 8000:127.0.0.1:8000 dgx-s-bmu-cse-230480@10.1.0.176
```

Then test on GTA PC:

```bash
curl http://127.0.0.1:8000/health
```

And set client config:

- `MODEL_HOST = "127.0.0.1"`
- `MODEL_PORT = 8000`

## 8) Current API response keys from `/segment`

- `shape`
- `overlay_png_base64`
- `semantic_id_png_base64`
- `road_mask_png_base64`
- `road_category_ids`
- `segments`
- `latency_ms`

For autonomous logic, prefer:

- `road_mask_png_base64` (binary drivable mask)
- `semantic_id_png_base64` (per-pixel class IDs)
