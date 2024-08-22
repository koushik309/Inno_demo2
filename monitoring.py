from biomass import get_status_and_recommendation
def generate_monitoring_info(biomass, sick_spots, week):
    status, recommendation = get_status_and_recommendation(biomass, sick_spots)
    return status, recommendation
