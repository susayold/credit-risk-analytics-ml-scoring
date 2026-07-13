/*
01_create_base_application.sql

Creates the labeled base application view used for analysis.
This CV version focuses on the 307K+ labeled training applications because
the user-facing project analysis and model evaluation use TARGET.
*/

CREATE OR REPLACE VIEW v01_application_base AS
SELECT
    a.*
FROM application_train a;

