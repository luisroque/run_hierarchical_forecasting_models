library('fpp3')
prison_results <- function(df) {
  prison <- df %>%
    mutate(Quarter = yearquarter(Date)) %>%
    select(-Date)  %>%
    as_tsibble(key = c(Gender, Legal, State), index = Quarter) %>%
    relocate(Quarter)
  
  prison_gts <- prison %>%
    aggregate_key(Gender * Legal * State, Count = sum(Count)/1e3)
  
  fit <- prison_gts %>%
    filter(year(Quarter) <= 2014) %>%
    model(base = ETS(Count)) %>%
    reconcile(
      bottom_up = bottom_up(base),
      MinT = min_trace(base, method = "mint_shrink")
    )
  fc <- fit %>% forecast(h = 8)
  
  fc_csv = fc %>% 
    as_tibble %>% 
    filter(.model=='MinT') %>% 
    select(-Count) %>% 
    mutate(.mean=.mean*1e3) %>%
    mutate(.mean=(sprintf("%0.2f", .mean))) %>%
    rename(time=Quarter) %>%
    lapply(as.character) %>% 
    data.frame(stringsAsFactors=FALSE)
  
  return (fc_csv)
}



