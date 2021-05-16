numeric_features = 	['Lot_Frontage',
					'Lot_Area',
					'Year_Built',
					'Year_Remod_Add',
					'Bsmt_Unf_SF',
					'Total_Bsmt_SF',
					'Bsmt_Unf_SF',
 					'Total_Bsmt_SF',
 					'First_Flr_SF',
					'Second_Flr_SF',
					'Gr_Liv_Area',
					'Bsmt_Full_Bath',
					'Full_Bath',
					'Bedroom_AbvGr',
					'Kitchen_AbvGr',
					'TotRms_AbvGrd',
					'Fireplaces',
					'Garage_Cars',
					'Garage_Area',
					'Open_Porch_SF',
					'Enclosed_Porch',
					'Screen_Porch',
					'Pool_Area',
					'Misc_Val',
					'Year_Sold']

boolean_feature = ['Central_Air']

categorical_nominal_features = 	['MS_SubClass',
								'MS_Zoning',
								'Street',
								'Alley',
								'Lot_Shape',
								'Land_Contour',
								'Utilities',
								'Bldg_Type',
								'House_Style',
								'Roof_Style',
								'Foundation',
								'Bsmt_Exposure',
								'Heating',
								'Electrical',
								'Functional',
								'Garage_Type',
								'Paved_Drive',
								'Fence',
 								'Misc_Feature',
 								'Mo_Sold',
								'Sale_Type',
								'Sale_Condition']

categorical_ordinal_features = 	['Overall_Qual',
								'Overall_Cond',
								'Exter_Qual',
								'Exter_Cond',
								'Bsmt_Qual',
								'Bsmt_Cond',
								'Heating_QC',
								'Kitchen_Qual',
								'Fireplace_Qu',
								'Garage_Qual',
								'Garage_Cond',
								'Pool_QC']

categorical_ordinal_features_mappers = 	[{'Very_Poor':0,'Poor':1,'Below_Average':2,'Average':3,'Above_Average':4,'Fair':5,'Good':6,'Very_Good':7,'Excellent':8,'Very_Excellent':9},
										{'Very_Poor':0,'Poor':1,'Below_Average':2,'Average':3,'Above_Average':4,'Fair':5,'Good':6,'Very_Good':7,'Excellent':8,'Very_Excellent':9},
										{'Poor':1,'Typical':3,'Fair':5,'Good':6,'Excellent':8},
										{'Poor':1,'Typical':3,'Fair':5,'Good':6,'Excellent':8},
										{'No_Basement':0,'Poor':1,'Typical':3,'Fair':5,'Good':6,'Excellent':8},
										{'No_Basement':0,'Poor':1,'Typical':3,'Fair':5,'Good':6,'Excellent':8},
										{'Poor':1,'Typical':3,'Fair':5,'Good':6,'Excellent':8},
										{'Poor':1,'Typical':3,'Fair':5,'Good':6,'Excellent':8},
										{'No_Fireplace':0,'Poor':1,'Typical':3,'Fair':5,'Good':6,'Excellent':8},
										{'No_Garage':0,'Poor':1,'Typical':3,'Fair':5,'Good':6,'Excellent':8},
										{'No_Garage':0,'Poor':1,'Typical':3,'Fair':5,'Good':6,'Excellent':8},
										{'No_Pool':0,'Poor':1,'Typical':3,'Fair':5,'Good':6,'Excellent':8}]
