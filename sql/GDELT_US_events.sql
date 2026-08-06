DECLARE start_date DATE DEFAULT DATE '2019-11-30';
DECLARE end_date DATE DEFAULT DATE '2021-05-15';

CREATE TEMP TABLE calendar AS
SELECT date
FROM UNNEST(GENERATE_DATE_ARRAY(start_date, end_date)) AS date;

CREATE TEMP TABLE us_events AS
SELECT
  PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) AS date,
  LPAD(CAST(EventRootCode AS STRING), 2, '0') AS root_code,
  SAFE_CAST(QuadClass AS INT64) AS quad_class,
  SAFE_CAST(IsRootEvent AS INT64) AS is_root_event,
  COALESCE(SAFE_CAST(NumMentions AS FLOAT64), 0) AS num_mentions,
  COALESCE(SAFE_CAST(NumSources AS FLOAT64), 0) AS num_sources,
  COALESCE(SAFE_CAST(NumArticles AS FLOAT64), 0) AS num_articles,
  SAFE_CAST(GoldsteinScale AS FLOAT64) AS goldstein_scale,
  SAFE_CAST(AvgTone AS FLOAT64) AS avg_tone
FROM `gdelt-bq.gdeltv2.events_partitioned`
WHERE _PARTITIONDATE BETWEEN start_date AND end_date
  AND SQLDATE BETWEEN CAST(FORMAT_DATE('%Y%m%d', start_date) AS INT64)
                  AND CAST(FORMAT_DATE('%Y%m%d', end_date) AS INT64)
  AND ActionGeo_CountryCode = 'US';

WITH daily AS (
  SELECT
    date,

    COUNT(*) AS us_event_count,
    COUNTIF(is_root_event = 1) AS us_root_event_count,
    SUM(num_mentions) AS us_event_mentions,
    SUM(num_sources) AS us_event_source_mentions,
    SUM(num_articles) AS us_event_articles,
    AVG(goldstein_scale) AS us_mean_goldstein,
    SAFE_DIVIDE(
      SUM(IF(goldstein_scale IS NOT NULL, goldstein_scale * num_mentions, 0)),
      SUM(IF(goldstein_scale IS NOT NULL, num_mentions, 0))
    ) AS us_mention_weighted_goldstein,
    AVG(avg_tone) AS us_mean_event_tone,
    SAFE_DIVIDE(
      SUM(IF(avg_tone IS NOT NULL, avg_tone * num_articles, 0)),
      SUM(IF(avg_tone IS NOT NULL, num_articles, 0))
    ) AS us_article_weighted_event_tone,
    SAFE_DIVIDE(COUNTIF(quad_class = 3), COUNT(*)) AS verbal_conflict_share,
    SAFE_DIVIDE(COUNTIF(quad_class = 4), COUNT(*)) AS material_conflict_share,

    COUNTIF(root_code = '02') AS appeal_event_count,
    COUNTIF(root_code = '02' AND is_root_event = 1) AS appeal_root_event_count,
    SAFE_DIVIDE(SUM(IF(root_code = '02', num_articles, 0)), SUM(num_articles)) AS appeal_article_share,
    SAFE_DIVIDE(
      SUM(IF(root_code = '02' AND avg_tone IS NOT NULL, avg_tone * num_articles, 0)),
      SUM(IF(root_code = '02' AND avg_tone IS NOT NULL, num_articles, 0))
    ) AS appeal_tone,

    COUNTIF(root_code = '09') AS investigation_event_count,
    COUNTIF(root_code = '09' AND is_root_event = 1) AS investigation_root_event_count,
    SAFE_DIVIDE(SUM(IF(root_code = '09', num_articles, 0)), SUM(num_articles)) AS investigation_article_share,
    SAFE_DIVIDE(
      SUM(IF(root_code = '09' AND avg_tone IS NOT NULL, avg_tone * num_articles, 0)),
      SUM(IF(root_code = '09' AND avg_tone IS NOT NULL, num_articles, 0))
    ) AS investigation_tone,

    COUNTIF(root_code = '10') AS demand_event_count,
    COUNTIF(root_code = '10' AND is_root_event = 1) AS demand_root_event_count,
    SAFE_DIVIDE(SUM(IF(root_code = '10', num_articles, 0)), SUM(num_articles)) AS demand_article_share,
    SAFE_DIVIDE(
      SUM(IF(root_code = '10' AND avg_tone IS NOT NULL, avg_tone * num_articles, 0)),
      SUM(IF(root_code = '10' AND avg_tone IS NOT NULL, num_articles, 0))
    ) AS demand_tone,

    COUNTIF(root_code = '13') AS threat_event_count,
    COUNTIF(root_code = '13' AND is_root_event = 1) AS threat_root_event_count,
    SAFE_DIVIDE(SUM(IF(root_code = '13', num_articles, 0)), SUM(num_articles)) AS threat_article_share,
    SAFE_DIVIDE(
      SUM(IF(root_code = '13' AND avg_tone IS NOT NULL, avg_tone * num_articles, 0)),
      SUM(IF(root_code = '13' AND avg_tone IS NOT NULL, num_articles, 0))
    ) AS threat_tone,

    COUNTIF(root_code = '14') AS protest_event_count,
    COUNTIF(root_code = '14' AND is_root_event = 1) AS protest_root_event_count,
    SAFE_DIVIDE(SUM(IF(root_code = '14', num_articles, 0)), SUM(num_articles)) AS protest_article_share,
    SAFE_DIVIDE(
      SUM(IF(root_code = '14' AND avg_tone IS NOT NULL, avg_tone * num_articles, 0)),
      SUM(IF(root_code = '14' AND avg_tone IS NOT NULL, num_articles, 0))
    ) AS protest_tone,

    COUNTIF(root_code = '17') AS coercion_event_count,
    COUNTIF(root_code = '17' AND is_root_event = 1) AS coercion_root_event_count,
    SAFE_DIVIDE(SUM(IF(root_code = '17', num_articles, 0)), SUM(num_articles)) AS coercion_article_share,
    SAFE_DIVIDE(
      SUM(IF(root_code = '17' AND avg_tone IS NOT NULL, avg_tone * num_articles, 0)),
      SUM(IF(root_code = '17' AND avg_tone IS NOT NULL, num_articles, 0))
    ) AS coercion_tone,

    COUNTIF(root_code = '18') AS assault_event_count,
    COUNTIF(root_code = '18' AND is_root_event = 1) AS assault_root_event_count,
    SAFE_DIVIDE(SUM(IF(root_code = '18', num_articles, 0)), SUM(num_articles)) AS assault_article_share,
    SAFE_DIVIDE(
      SUM(IF(root_code = '18' AND avg_tone IS NOT NULL, avg_tone * num_articles, 0)),
      SUM(IF(root_code = '18' AND avg_tone IS NOT NULL, num_articles, 0))
    ) AS assault_tone,

    COUNTIF(root_code = '19') AS fighting_event_count,
    COUNTIF(root_code = '19' AND is_root_event = 1) AS fighting_root_event_count,
    SAFE_DIVIDE(SUM(IF(root_code = '19', num_articles, 0)), SUM(num_articles)) AS fighting_article_share,
    SAFE_DIVIDE(
      SUM(IF(root_code = '19' AND avg_tone IS NOT NULL, avg_tone * num_articles, 0)),
      SUM(IF(root_code = '19' AND avg_tone IS NOT NULL, num_articles, 0))
    ) AS fighting_tone,

    COUNTIF(root_code = '20') AS mass_violence_event_count,
    COUNTIF(root_code = '20' AND is_root_event = 1) AS mass_violence_root_event_count,
    SAFE_DIVIDE(SUM(IF(root_code = '20', num_articles, 0)), SUM(num_articles)) AS mass_violence_article_share,
    SAFE_DIVIDE(
      SUM(IF(root_code = '20' AND avg_tone IS NOT NULL, avg_tone * num_articles, 0)),
      SUM(IF(root_code = '20' AND avg_tone IS NOT NULL, num_articles, 0))
    ) AS mass_violence_tone,

    COUNTIF(root_code IN ('13', '14', '17', '18', '19', '20')) AS contentious_event_count,
    SAFE_DIVIDE(
      COUNTIF(root_code IN ('13', '14', '17', '18', '19', '20')),
      COUNT(*)
    ) AS contentious_event_share,
    SAFE_DIVIDE(
      SUM(IF(root_code IN ('13', '14', '17', '18', '19', '20') AND goldstein_scale IS NOT NULL, goldstein_scale * num_mentions, 0)),
      SUM(IF(root_code IN ('13', '14', '17', '18', '19', '20') AND goldstein_scale IS NOT NULL, num_mentions, 0))
    ) AS contentious_mention_weighted_goldstein,

    COUNTIF(root_code IN ('18', '19', '20')) AS violence_event_count,
    SAFE_DIVIDE(
      COUNTIF(root_code IN ('18', '19', '20')),
      COUNT(*)
    ) AS violence_event_share,
    SAFE_DIVIDE(
      SUM(IF(root_code IN ('18', '19', '20') AND goldstein_scale IS NOT NULL, goldstein_scale * num_mentions, 0)),
      SUM(IF(root_code IN ('18', '19', '20') AND goldstein_scale IS NOT NULL, num_mentions, 0))
    ) AS violence_mention_weighted_goldstein
  FROM us_events
  GROUP BY date
)
SELECT
  calendar.date,
  COALESCE(daily.us_event_count, 0) AS us_event_count,
  COALESCE(daily.us_root_event_count, 0) AS us_root_event_count,
  COALESCE(daily.us_event_mentions, 0) AS us_event_mentions,
  COALESCE(daily.us_event_source_mentions, 0) AS us_event_source_mentions,
  COALESCE(daily.us_event_articles, 0) AS us_event_articles,
  daily.* EXCEPT (
    date,
    us_event_count,
    us_root_event_count,
    us_event_mentions,
    us_event_source_mentions,
    us_event_articles
  )
FROM calendar
LEFT JOIN daily USING (date)
ORDER BY date;
