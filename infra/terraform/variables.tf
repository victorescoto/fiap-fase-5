variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "passos-magicos-api"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "enable_apprunner" {
  type    = bool
  default = true
}

variable "dashboard_name" {
  type    = string
  default = "passos-magicos-dashboard"
}