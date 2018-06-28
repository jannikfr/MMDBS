create table mmdbs_image
(
  id                    serial       not null
    constraint features_pkey
    primary key,
  path                  varchar(255) not null,
  classification        varchar(255),
  local_histogram1      json,
  global_histogram      json,
  global_edge_histogram json,
  local_histogram2      json,
  local_histogram3      json,
  global_hue_histogram json,
);

