DROP TABLE IF EXISTS [FullCompanyList];
DROP TABLE IF EXISTS [ProgramTracker];
DROP TABLE IF EXISTS [MappingHarvest];
DROP TABLE IF EXISTS [MarketTDCompanies];
DROP TABLE IF EXISTS [HubSpotCompanyDetails];
DROP TABLE IF EXISTS [MainPrograms];



CREATE TABLE FullCompanyList(
  "Steve notes" TEXT,
  "Company_Name" VarChar(500),
  "Industry Suggestion (Old Categories)" TEXT,
  "Target Vertical" TEXT,
  "Sub-Vertical" TEXT,
  "Business Type" TEXT,
  "Sales Model" TEXT,
  "B2B Domain" TEXT,
   CONSTRAINT [PK_FullCompanyList] PRIMARY KEY  ([Company_Name])

);

CREATE TABLE ProgramTracker(
  "Program_UID" INT,
  "Program" TEXT,
  "Client" TEXT,
  "Company Name (Target/PortCo)" TEXT,
  "Program_Type" TEXT,
  "Industry" TEXT,
  "Program Start Date" TEXT,
  "Program End Date" TEXT,
  "SOW Link" TEXT,
  "Final Deliverable Link" TEXT,
  CONSTRAINT [PK_ProgramTracker] PRIMARY KEY  ([Program_UID])

);

CREATE TABLE MainPrograms(
  "Program_UID" INT,
  "Harvest_Project" TEXT,
  "Deal_Category" TEXT,
  "Deal_Amount" TEXT,
  "Client" TEXT,
  "Industry" TEXT,
  "Program_Status" TEXT,
  "Program_Start_Date" TEXT,
  "Program_End_Date" TEXT,
  "SOW_Sign_Date" TEXT,
  "Final_Deliverable_Link" TEXT,
  CONSTRAINT [PK_MainPrograms] PRIMARY KEY  ([Program_UID])  
);

CREATE TABLE HubSpotCompanyDetails(
  "ID" INT,
  "Hubspot ID" TEXT,
  "Company_name" TEXT,
  "Active Investors" TEXT,
  "Total Funding" TEXT,
  "Number of Employees" TEXT,
  "Industry" TEXT,
  "Annual Revenue" TEXT,
  "ev_Industry" TEXT,
  CONSTRAINT [PK_HubSpotCompanyDetails] PRIMARY KEY  ([ID])
);

CREATE TABLE MappingHarvest(
  "harvest_id" TEXT,
  "harvest_project_name" VarChar(500),
  "harvest_client" TEXT,
  "hubspot_match" VarChar(500),
  "Program_Tracker_ID" INT,
  "Program_UID"  INT,
  CONSTRAINT [PK_MappingHarvest] PRIMARY KEY  ([harvest_id]),
  FOREIGN KEY ([hubspot_match]) REFERENCES [HubSpotCompanyDetails] ([Company_name]),
  FOREIGN KEY ([Program_Tracker_ID]) REFERENCES [ProgramTracker] ([Program_UID]),
  FOREIGN KEY ([Program_UID]) REFERENCES [MainPrograms] ([Program_UID])
);


CREATE TABLE HubSpotContacts (
    id INTEGER,
    "First Name" TEXT,
    "Last Name" TEXT,
    "Email Domain" TEXT,
    Email TEXT,
    "Mobile Phone Number" TEXT,
    "Job Title" TEXT,
    Name TEXT,
    "Company_IDs" TEXT
);


CREATE TABLE HubSpotContactCompanies(
    id INTEGER,
    Company_IDs INTEGER,
    FOREIGN KEY ([id]) REFERENCES [HubSpotContacts] ([id])
    FOREIGN KEY ([Company_IDs]) REFERENCES [HubSpotCompanyDetails] (["Hubspot ID"])
);


.mode csv
.import csv_files/FinalMarketTDCompanies.csv MarketTDCompanies
.import csv_files/FinalMainPrograms.csv MainPrograms
.import csv_files/FinalHubSpotCompanyDetails.csv HubSpotCompanyDetails
.import csv_files/FinalMappingHarvest.csv MappingHarvest
.import csv_files/FinalProgramTracker.csv ProgramTracker
.import csv_files/FinalHubSpotContacts.csv HubSpotContacts 
.import csv_files/FinalHubSpotContactCompanies.csv HubSpotContactCompanies 

/** MARKET TD COMPANIES HAS NOT BEEN UPDATED**/

/** INSERT COLUMNS **/
-- ALTER TABLE MappingHarvest ADD COLUMN company_details_name VarChar(500) REFERENCES HubSpotCompanyDetails("Company name");
-- ALTER TABLE MappingHarvest ADD COLUMN main_program_uid INT REFERENCES MainPrograms(Program_UID);
-- ALTER TABLE MappingHarvest ADD COLUMN program_tracker_uid INT REFERENCES ProgramTracker(Program_UID);

-- /** UPDATE VALUES **/
-- UPDATE MappingHarvest
-- SET 
--     company_details_name = hubspot_match,
--     main_program_uid = Program_UID,
--     program_tracker_uid = Program_Tracker_ID;

/** RESOLVE COULMNS WITH BLANK NAMES**/
ALTER TABLE HubSpotCompanyDetails RENAME COLUMN "" TO ev_updated_empty_column;
ALTER TABLE MainPrograms RENAME COLUMN "" TO ev_updated_empty_column;
ALTER TABLE ProgramTracker RENAME COLUMN "" TO ev_updated_empty_column;
ALTER TABLE MarketTDCompanies RENAME COLUMN "" TO ev_updated_empty_column;
ALTER TABLE MappingHarvest RENAME COLUMN "" TO ev_updated_empty_column;
ALTER TABLE FullCompanyList RENAME COLUMN "" TO ev_updated_empty_column;

