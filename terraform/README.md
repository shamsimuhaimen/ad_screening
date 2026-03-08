# Terraform Setup

Infra in this directory provisions:
- `vscode-cpu` Compute Engine VM (`e2-standard-8`)
- `perma-disk` 200GB persistent SSD mounted at `/mnt/perma`

Boot disk is 30GB `pd-standard` (can align with free-tier standard disk allowance).

## Prerequisites
- Terraform >= 1.3
- `gcloud` authenticated to the correct account/project
- APIs enabled in `ai-drug-discovery-uoft-2026`:
  - `compute.googleapis.com`

## Apply
```bash
cd terraform
terraform init
terraform plan
terraform apply
```

## Connect
Use OS Login:
```bash
gcloud compute ssh vscode-cpu --zone northamerica-northeast2-a --project ai-drug-discovery-uoft-2026
```

## Destroy
`perma-disk` has `prevent_destroy = true`, so remove it intentionally only when needed.

To destroy VM only:
```bash
cd terraform
terraform destroy -target=google_compute_instance.vscode_cpu
```

To destroy disk too, first remove/override `prevent_destroy`, then destroy.
