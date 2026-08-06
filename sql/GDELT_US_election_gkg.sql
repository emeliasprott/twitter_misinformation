DECLARE start_date DATE DEFAULT DATE '2019-11-30';
DECLARE end_date DATE DEFAULT DATE '2021-05-15';
DECLARE include_translated BOOL DEFAULT FALSE;

CREATE TEMP TABLE calendar AS
SELECT date
FROM UNNEST(GENERATE_DATE_ARRAY(start_date, end_date)) AS date;

CREATE TEMP TABLE gkg_us_docs AS
WITH raw AS (
  SELECT
    _PARTITIONDATE AS date,
    COALESCE(NULLIF(SourceCommonName, ''), 'unknown') AS source,
    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS tone,
    CONCAT(';', UPPER(REGEXP_REPLACE(IFNULL(V2Themes, ''), r',\d+', '')), ';') AS themes,
    IFNULL(V2Locations, '') AS locations,
    LOWER(REGEXP_REPLACE(IFNULL(V2Persons, ''), r',\d+', '')) AS persons,
    LOWER(REGEXP_REPLACE(IFNULL(V2Organizations, ''), r',\d+', '')) AS organizations
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE _PARTITIONDATE BETWEEN start_date AND end_date
    AND SourceCollectionIdentifier = 1
    AND (include_translated OR IFNULL(TranslationInfo, '') = '')
),
primitive_flags AS (
  SELECT
    *,
    REGEXP_CONTAINS(locations, r'(^|;)[1-5]#[^#;]*#US#') AS has_us_location,

    REGEXP_CONTAINS(
      persons,
      r'(donald( j)? trump|joe biden|joseph( r)? biden|mike pence|kamala harris|rudy giuliani|sidney powell)'
    ) AS has_us_election_actor,

    REGEXP_CONTAINS(
      organizations,
      r'(republican party|democratic party|republican national committee|democratic national committee|united states congress|u\.s\. congress|white house|supreme court|department of justice|federal election commission|election assistance commission|board of elections|electoral college)'
    ) AS has_us_political_institution,

    STRPOS(themes, ';ELECTION;') > 0 AS has_election_theme,

    REGEXP_CONTAINS(
      themes,
      r';(GENERAL_GOVERNMENT|LEGISLATION|WB_831_GOVERNANCE|DELAY);'
    ) AS has_election_administration_theme,

    REGEXP_CONTAINS(
      themes,
      r';(WB_840_JUSTICE|WB_1014_CRIMINAL_JUSTICE|TRIAL|EPU_POLICY_LAW);'
    ) AS has_election_litigation_theme,

    REGEXP_CONTAINS(
      themes,
      r';(WB_832_ANTI_CORRUPTION|WB_2024_ANTI_CORRUPTION_AUTHORITIES|SOC_GENERALCRIME|SECURITY_SERVICES|ARREST);'
    ) AS has_integrity_security_theme,

    REGEXP_CONTAINS(
      themes,
      r';(PROTEST|ARMEDCONFLICT|WB_2432_FRAGILITY_CONFLICT_AND_VIOLENCE|WB_2433_CONFLICT_AND_VIOLENCE|KILL|MANMADE_DISASTER_IMPLIED);'
    ) AS has_mobilization_conflict_theme,

    REGEXP_CONTAINS(
      themes,
      r';(MEDIA_SOCIAL|MEDIA_MSM|WB_133_INFORMATION_AND_COMMUNICATION_TECHNOLOGIES|WB_694_BROADCAST_AND_MEDIA|WB_678_DIGITAL_GOVERNMENT);'
    ) AS has_media_platform_theme,

    REGEXP_CONTAINS(persons, r'donald( j)? trump') AS mentions_trump,
    REGEXP_CONTAINS(persons, r'(joe biden|joseph( r)? biden)') AS mentions_biden,

    REGEXP_CONTAINS(
      organizations,
      r'(republican party|republican national committee)'
    ) AS mentions_republican_party,

    REGEXP_CONTAINS(
      organizations,
      r'(democratic party|democratic national committee)'
    ) AS mentions_democratic_party,

    REGEXP_CONTAINS(
      organizations,
      r'(federal election commission|election assistance commission|board of elections|secretary of state|united states postal service|u\.s\. postal service|electoral college|state legislature)'
    ) AS mentions_election_administration,

    REGEXP_CONTAINS(
      organizations,
      r'(supreme court|court of appeals|district court|department of justice)'
    ) AS mentions_court_or_justice,

    REGEXP_CONTAINS(
      organizations,
      r'(dominion voting systems|smartmatic|election systems & software|election systems and software)'
    ) AS mentions_election_technology,

    REGEXP_CONTAINS(
      organizations,
      r'(twitter|facebook|youtube|google)'
    ) AS mentions_major_platform,

    REGEXP_CONTAINS(
      organizations,
      r'(proud boys|oath keepers|three percenters|3 percenters|boogaloo)'
    ) AS mentions_extremist_organization
  FROM raw
),
flags AS (
  SELECT
    *,
    (
      has_us_location
      OR has_us_election_actor
      OR has_us_political_institution
    ) AS is_us_context,

    has_election_theme AS is_election,

    (
      has_election_theme
      AND (mentions_election_administration OR has_election_administration_theme)
    ) AS is_election_administration,

    (
      has_election_theme
      AND (mentions_court_or_justice OR has_election_litigation_theme)
    ) AS is_election_litigation,

    (
      has_election_theme
      AND (has_integrity_security_theme OR mentions_election_technology)
    ) AS is_election_integrity_security,

    (
      has_election_theme
      AND mentions_election_technology
    ) AS is_election_technology,

    (
      has_election_theme
      AND (has_mobilization_conflict_theme OR mentions_extremist_organization)
    ) AS is_election_mobilization,

    (
      has_election_theme
      AND (has_media_platform_theme OR mentions_major_platform)
    ) AS is_election_platform,

    (has_election_theme AND mentions_trump) AS is_trump_election,
    (has_election_theme AND mentions_biden) AS is_biden_election,
    (has_election_theme AND mentions_republican_party) AS is_republican_election,
    (has_election_theme AND mentions_democratic_party) AS is_democratic_election
  FROM primitive_flags
)
SELECT
  date,
  source,
  tone,
  is_election,
  is_election_administration,
  is_election_litigation,
  is_election_integrity_security,
  is_election_technology,
  is_election_mobilization,
  is_election_platform,
  mentions_trump,
  mentions_biden,
  mentions_republican_party,
  mentions_democratic_party,
  is_trump_election,
  is_biden_election,
  is_republican_election,
  is_democratic_election
FROM flags
WHERE is_us_context;

WITH daily AS (
  SELECT
    date,

    COUNT(*) AS us_document_count,
    APPROX_COUNT_DISTINCT(source) AS us_source_count,
    AVG(tone) AS us_mean_tone,
    STDDEV_SAMP(tone) AS us_tone_sd,
    SAFE_DIVIDE(COUNTIF(tone < 0), COUNTIF(tone IS NOT NULL)) AS us_negative_share,
    SAFE_DIVIDE(COUNTIF(tone > 0), COUNTIF(tone IS NOT NULL)) AS us_positive_share,

    COUNTIF(is_election) AS election_document_count,
    SAFE_DIVIDE(COUNTIF(is_election), COUNT(*)) AS election_share,
    APPROX_COUNT_DISTINCT(IF(is_election, source, NULL)) AS election_source_count,
    AVG(IF(is_election, tone, NULL)) AS election_tone,
    AVG(IF(is_election AND tone < 0, -tone, NULL)) AS election_negative_intensity,

    COUNTIF(is_election_administration) AS election_administration_document_count,
    SAFE_DIVIDE(COUNTIF(is_election_administration), COUNT(*)) AS election_administration_share,
    APPROX_COUNT_DISTINCT(IF(is_election_administration, source, NULL)) AS election_administration_source_count,
    AVG(IF(is_election_administration, tone, NULL)) AS election_administration_tone,
    AVG(IF(is_election_administration AND tone < 0, -tone, NULL)) AS election_administration_negative_intensity,

    COUNTIF(is_election_litigation) AS election_litigation_document_count,
    SAFE_DIVIDE(COUNTIF(is_election_litigation), COUNT(*)) AS election_litigation_share,
    APPROX_COUNT_DISTINCT(IF(is_election_litigation, source, NULL)) AS election_litigation_source_count,
    AVG(IF(is_election_litigation, tone, NULL)) AS election_litigation_tone,
    AVG(IF(is_election_litigation AND tone < 0, -tone, NULL)) AS election_litigation_negative_intensity,

    COUNTIF(is_election_integrity_security) AS election_integrity_security_document_count,
    SAFE_DIVIDE(COUNTIF(is_election_integrity_security), COUNT(*)) AS election_integrity_security_share,
    APPROX_COUNT_DISTINCT(IF(is_election_integrity_security, source, NULL)) AS election_integrity_security_source_count,
    AVG(IF(is_election_integrity_security, tone, NULL)) AS election_integrity_security_tone,
    AVG(IF(is_election_integrity_security AND tone < 0, -tone, NULL)) AS election_integrity_security_negative_intensity,

    COUNTIF(is_election_technology) AS election_technology_document_count,
    SAFE_DIVIDE(COUNTIF(is_election_technology), COUNT(*)) AS election_technology_share,
    APPROX_COUNT_DISTINCT(IF(is_election_technology, source, NULL)) AS election_technology_source_count,
    AVG(IF(is_election_technology, tone, NULL)) AS election_technology_tone,
    AVG(IF(is_election_technology AND tone < 0, -tone, NULL)) AS election_technology_negative_intensity,

    COUNTIF(is_election_mobilization) AS election_mobilization_document_count,
    SAFE_DIVIDE(COUNTIF(is_election_mobilization), COUNT(*)) AS election_mobilization_share,
    APPROX_COUNT_DISTINCT(IF(is_election_mobilization, source, NULL)) AS election_mobilization_source_count,
    AVG(IF(is_election_mobilization, tone, NULL)) AS election_mobilization_tone,
    AVG(IF(is_election_mobilization AND tone < 0, -tone, NULL)) AS election_mobilization_negative_intensity,

    COUNTIF(is_election_platform) AS election_platform_document_count,
    SAFE_DIVIDE(COUNTIF(is_election_platform), COUNT(*)) AS election_platform_share,
    APPROX_COUNT_DISTINCT(IF(is_election_platform, source, NULL)) AS election_platform_source_count,
    AVG(IF(is_election_platform, tone, NULL)) AS election_platform_tone,
    AVG(IF(is_election_platform AND tone < 0, -tone, NULL)) AS election_platform_negative_intensity,

    COUNTIF(mentions_trump) AS trump_document_count,
    SAFE_DIVIDE(COUNTIF(mentions_trump), COUNT(*)) AS trump_share,
    AVG(IF(mentions_trump, tone, NULL)) AS trump_tone,

    COUNTIF(mentions_biden) AS biden_document_count,
    SAFE_DIVIDE(COUNTIF(mentions_biden), COUNT(*)) AS biden_share,
    AVG(IF(mentions_biden, tone, NULL)) AS biden_tone,

    COUNTIF(mentions_republican_party) AS republican_party_document_count,
    SAFE_DIVIDE(COUNTIF(mentions_republican_party), COUNT(*)) AS republican_party_share,
    AVG(IF(mentions_republican_party, tone, NULL)) AS republican_party_tone,

    COUNTIF(mentions_democratic_party) AS democratic_party_document_count,
    SAFE_DIVIDE(COUNTIF(mentions_democratic_party), COUNT(*)) AS democratic_party_share,
    AVG(IF(mentions_democratic_party, tone, NULL)) AS democratic_party_tone,

    COUNTIF(is_trump_election) AS trump_election_document_count,
    SAFE_DIVIDE(COUNTIF(is_trump_election), COUNT(*)) AS trump_election_share,
    AVG(IF(is_trump_election, tone, NULL)) AS trump_election_tone,

    COUNTIF(is_biden_election) AS biden_election_document_count,
    SAFE_DIVIDE(COUNTIF(is_biden_election), COUNT(*)) AS biden_election_share,
    AVG(IF(is_biden_election, tone, NULL)) AS biden_election_tone,

    COUNTIF(is_republican_election) AS republican_election_document_count,
    SAFE_DIVIDE(COUNTIF(is_republican_election), COUNT(*)) AS republican_election_share,
    AVG(IF(is_republican_election, tone, NULL)) AS republican_election_tone,

    COUNTIF(is_democratic_election) AS democratic_election_document_count,
    SAFE_DIVIDE(COUNTIF(is_democratic_election), COUNT(*)) AS democratic_election_share,
    AVG(IF(is_democratic_election, tone, NULL)) AS democratic_election_tone
  FROM gkg_us_docs
  GROUP BY date
)
SELECT
  calendar.date,
  COALESCE(daily.us_document_count, 0) AS us_document_count,
  COALESCE(daily.us_source_count, 0) AS us_source_count,
  daily.us_mean_tone,
  daily.us_tone_sd,
  COALESCE(daily.us_negative_share, 0) AS us_negative_share,
  COALESCE(daily.us_positive_share, 0) AS us_positive_share,
  daily.* EXCEPT (
    date,
    us_document_count,
    us_source_count,
    us_mean_tone,
    us_tone_sd,
    us_negative_share,
    us_positive_share
  )
FROM calendar
LEFT JOIN daily USING (date)
ORDER BY date;
