# infrastructure/glue_taxi.tf
resource "aws_glue_catalog_table" "yellow_taxi" {
  database_name = aws_glue_catalog_database.main.name
  name          = "yellow_taxi"
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    "classification"            = "parquet"
    "parquet.compress"          = "SNAPPY"
    "projection.enabled"        = "true"
    "projection.year.type"      = "integer"
    "projection.year.range"     = "2019,2022"
    "projection.month.type"     = "integer"
    "projection.month.range"    = "1,12"
    "projection.month.digits"   = "2"
    "storage.location.template" = "s3://${aws_s3_bucket.workspace.bucket}/data/taxi/year=$${year}/month=$${month}/"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.workspace.bucket}/data/taxi/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }
    columns {
      name = "pickup_date"
      type = "date"
    }
    columns {
      name = "pickup_hour"
      type = "int"
    }
    columns {
      name = "day_of_week"
      type = "int"
    }
    columns {
      name = "time_of_day"
      type = "string"
    }
    columns {
      name = "PULocationID"
      type = "int"
    }
    columns {
      name = "DOLocationID"
      type = "int"
    }
    columns {
      name = "trip_distance"
      type = "double"
    }
    columns {
      name = "distance_bucket"
      type = "string"
    }
    columns {
      name = "fare_amount"
      type = "double"
    }
    columns {
      name = "fare_norm"
      type = "double"
    }
    columns {
      name = "tip_amount"
      type = "double"
    }
    columns {
      name = "pay_credit_card"
      type = "int"
    }
    columns {
      name = "pay_cash"
      type = "int"
    }
  }

  partition_keys {
    name = "year"
    type = "int"
  }
  partition_keys {
    name = "month"
    type = "int"
  }
}
