output "instance_name" {
  description = "Name of the CPU instance."
  value       = google_compute_instance.vscode_cpu.name
}

output "instance_public_ip" {
  description = "Ephemeral external IPv4 address."
  value       = try(google_compute_instance.vscode_cpu.network_interface[0].access_config[0].nat_ip, null)
}

output "ssh_command" {
  description = "gcloud command for connecting with OS Login."
  value       = format("gcloud compute ssh %s --zone %s --project %s", google_compute_instance.vscode_cpu.name, var.zone, var.project_id)
}
