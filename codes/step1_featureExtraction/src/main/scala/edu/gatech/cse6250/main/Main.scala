/**
 * @author Hang Su <hangsu@gatech.edu>.
 * @author Yu Jing <yujing@gatech.edu>
 */

package edu.gatech.cse6250.main

import java.text.SimpleDateFormat

import edu.gatech.cse6250.helper.{ CSVHelper, SparkHelper }
import edu.gatech.cse6250.model._
import edu.gatech.cse6250.randomwalk.RandomWalk
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame

object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.{ Level, Logger }

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkHelper.spark
    val sc = spark.sparkContext

    /** initialize loading of data */
    loadRddRawData(spark)

    //val patientGraph = GraphLoader.load(patient, labResult, medication, diagnostic)

    sc.stop()
  }

  def loadRddRawData(spark: SparkSession): DataFrame = {
    import spark.implicits._
    val sqlContext = spark.sqlContext

    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")

    //List("data/files/mimiciii/1.4/PATIENTS.csv", "data/files/mimiciii/1.4/ICUSTAYS.csv",
    //  "data/files/mimiciii/1.4/CHARTEVENTS.csv", "data/files/mimiciii/1.4/LABEVENTS.csv")
    //  .foreach(CSVHelper.loadCSVAsTable(spark, _))

    List("data/files/mimiciii/1.4/PATIENTS.csv", "data/files/mimiciii/1.4/ADMISSIONS.csv",
      "data/files/mimiciii/1.4/DIAGNOSES_ICD.csv", "data/files/mimiciii/1.4/SERVICES.csv",
      "data/files/mimiciii/1.4/LABEVENTS.csv", "data/files/mimiciii/1.4/ICUSTAYS.csv",
      "data/files/mimiciii/1.4/OUTPUTEVENTS.csv")
      .foreach(CSVHelper.loadCSVAsTable(spark, _))
    // add logic to handle null values if needed

    val patients = sqlContext.sql(
      """
        |SELECT subject_id, gender, dob, dod, expire_flag
        |FROM PATIENTS
      """.stripMargin)
      .filter(x => (x != null && x.getString(0) != null))
      .na.fill("")
    patients.coalesce(1).write.option("header", "true").option("sep", ",").mode("overwrite").csv("data/feature_csv/patients")

    // ALL adults ICU stays & stay > 2 days & first icu stays
    // it includes AGE
    val icustays = sqlContext.sql(
       """
       |WITH co AS
       |(
       |SELECT icu.subject_id, icu.hadm_id, icu.icustay_id, icu.intime
       |, (UNIX_TIMESTAMP(outtime) - UNIX_TIMESTAMP(intime))/60.0/60.0/24.0 as icu_length_of_stay
       |, (UNIX_TIMESTAMP(icu.intime) - UNIX_TIMESTAMP(pat.dob))/ 60.0 / 60.0 / 24.0 / 365.242 as age
       | , RANK() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime) AS icustay_id_order
       |FROM ICUSTAYS icu INNER JOIN PATIENTS pat ON
       |  icu.subject_id = pat.subject_id
       |)
       |SELECT
       |FROM co WHERE age >=16 AND icu_length_of_stay >= 2 AND co.icustay_id_order = 1
       """.stripMargin)
    icustays.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("data/feature_csv/icustays")

    // heartrate
    val hr = sqlContext.sql(
       """
       SELECT subject_id, hadm_id, icustay_id, charttime, value from CHARTEVENTS WHERE itemid = 211 OR itemid = 220045
       """.stripMargin)
    hr.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("data/feature_csv/hr")

    val gcs = sqlContext.sql(
       """
       |with base as
       |(
       |select ce.icustay_id, ce.charttime
       |-- pivot each value into its own column
       |, max(case when ce.ITEMID in (454,223901) then ce.valuenum else null end) as GCSMotor
       |, max(case
       |when ce.ITEMID = 723 and ce.VALUE = '1.0 ET/Trach' then 0
       |when ce.ITEMID = 223900 and ce.VALUE = 'No Response-ETT' then 0
       |when ce.ITEMID in (723,223900) then ce.valuenum
       |else null
       |end) as GCSVerbal
       |, max(case when ce.ITEMID in (184,220739) then ce.valuenum else null end) as GCSEyes
       -- convert the data into a number, reserving a value of 0 for ET/Trach
       |, max(case
       -- endotrach/vent is assigned a value of 0, later parsed specially
       |when ce.ITEMID = 723 and ce.VALUE = '1.0 ET/Trach' then 1 -- carevue
       |when ce.ITEMID = 223900 and ce.VALUE = 'No Response-ETT' then 1 -- metavision
       |else 0 end)
       |as endotrachflag
       |, ROW_NUMBER ()
       |OVER (PARTITION BY ce.icustay_id ORDER BY ce.charttime ASC) as rn
       |from CHARTEVENTS ce
       |-- Isolate the desired GCS variables
       |where ce.ITEMID in
       |(
       |-- 198 -- GCS
       |-- GCS components, CareVue
       |184, 454, 723
       |-- GCS components, Metavision
       |, 223900, 223901, 220739
       |)
       |-- exclude rows marked as error
       |and ce.error IS DISTINCT FROM 1
       |group by ce.ICUSTAY_ID, ce.charttime
       |)
       |, gcs as (
       |select b.
       |, b2.GCSVerbal as GCSVerbalPrev
       |, b2.GCSMotor as GCSMotorPrev
       |, b2.GCSEyes as GCSEyesPrev
       |-- Calculate GCS, factoring in special case when they are intubated and prev vals
       |-- note that the coalesce are used to implement the following if:
       |--  if current value exists, use it
       |--  if previous value exists, use it
       |--  otherwise, default to normal
       |, case
       |-- replace GCS during sedation with 15
       |when b.GCSVerbal = 0
       |then 15
       |when b.GCSVerbal is null and b2.GCSVerbal = 0
       |then 15
       |-- if previously they were intub, but they aren't now, do not use previous GCS values
       |when b2.GCSVerbal = 0
       |then
       |coalesce(b.GCSMotor,6)
       |+ coalesce(b.GCSVerbal,5)
       |+ coalesce(b.GCSEyes,4)
       |-- otherwise, add up score normally, imputing previous value if none available at current time
       |else
       |coalesce(b.GCSMotor,coalesce(b2.GCSMotor,6))
       |+ coalesce(b.GCSVerbal,coalesce(b2.GCSVerbal,5))
       |+ coalesce(b.GCSEyes,coalesce(b2.GCSEyes,4))
       |end as GCS

       |from base b
       |-- join to itself within 6 hours to get previous value
       |left join base b2
       |on b.ICUSTAY_ID = b2.ICUSTAY_ID
       |and b.rn = b2.rn+1
       |and b2.charttime > b.charttime - interval '6' hour
       |)
       |-- combine components with previous within 6 hours
       |-- filter down to cohort which is not excluded
       |-- truncate charttime to the hour
       |, gcs_stg as
       |(
       |select gs.icustay_id, gs.charttime
       |, GCS
       |, coalesce(GCSMotor,GCSMotorPrev) as GCSMotor
       |, coalesce(GCSVerbal,GCSVerbalPrev) as GCSVerbal
       |, coalesce(GCSEyes,GCSEyesPrev) as GCSEyes
       |, case when coalesce(GCSMotor,GCSMotorPrev) is null then 0 else 1 end
       |+ case when coalesce(GCSVerbal,GCSVerbalPrev) is null then 0 else 1 end
       |+ case when coalesce(GCSEyes,GCSEyesPrev) is null then 0 else 1 end
       |as components_measured
       |, EndoTrachFlag
       |from gcs gs
       |)
       |-- priority is:
       |--  (i) complete data, (ii) non-sedated GCS, (iii) lowest GCS, (iv) charttime
       |, gcs_priority as
       |(
       |select icustay_id
       |, charttime
       |, GCS
       |, GCSMotor
       |, GCSVerbal
       |, GCSEyes
       |, EndoTrachFlag
       |, ROW_NUMBER() over
       |(
       |PARTITION BY icustay_id, charttime
       |ORDER BY components_measured DESC, endotrachflag, gcs, charttime DESC
       |) as rn
       |from gcs_stg
       |)
       |select icustay_id
       |, charttime
       |, GCS
       |, GCSMotor
       |, GCSVerbal
       |, GCSEyes
       |, EndoTrachFlag
       |from gcs_priority gs
       |where rn = 1
       |ORDER BY icustay_id, charttime""".stripMargin)
    gcs.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("data/feature_csv/gcs")

    // blood pressure
    val bp = sqlContext.sql(
       """
       |SELECT subject_id, hadm_id, icustay_id, charttime, value from CHARTEVENTS WHERE itemid = 51
       | OR itemid = 442 OR itemid = 455 OR itemid = 6701 OR itemid = 220179 OR itemid = 220050
       """.stripMargin)
    bp.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("data/feature_csv/blood_pressure")

    // body temperature
    val bt = sqlContext.sql(
       """
       |SELECT subject_id, hadm_id, icustay_id, charttime, value from CHARTEVENTS WHERE itemid = 678
       | OR itemid = 223761 OR itemid = 676 OR itemid = 223762
       """.stripMargin)
    bt.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("data/feature_csv/body_temperature")

    // pao2
    val pao2 = sqlContext.sql(
       """
       |SELECT subject_id, hadm_id, charttime, value from LABEVENTS WHERE itemid = 50821
       | OR itemid = 50816
       """.stripMargin)
    pao2.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("data/feature_csv/pao2")

    // fio2
    val fio2 = sqlContext.sql(
       """
       |SELECT subject_id, hadm_id, charttime, value from CHARTEVENTS WHERE itemid = 223835
       | OR itemid = 3420 OR itemid = 3422 OR itemid = 190
       """.stripMargin)
    fio2.coalesce(1).write.option("header","true").option("sep",",").mode("overwrite").csv("data/feature_csv/fio2")


    // next round
    // urine output
    val urine = sqlContext.sql(
      """
        |SELECT subject_id, hadm_id, icustay_id, charttime, value from OUTPUTEVENTS WHERE itemid = 40055
        | OR itemid = 43175 OR itemid = 40069 OR itemid = 40094 OR itemid = 40715 OR itemid = 40473
        | OR itemid = 40085 OR itemid = 40057 OR itemid = 40056 OR itemid = 40405 OR itemid = 40428
        | OR itemid = 40086 OR itemid = 40096 OR itemid = 40651 OR itemid = 226559 OR itemid = 226560
        | OR itemid = 226561 OR itemid = 226584 OR itemid = 226563 OR itemid = 226564 OR itemid = 226565
        | OR itemid = 226567 OR itemid = 226557 OR itemid = 226558 OR itemid = 227488 OR itemid = 227489
        """.stripMargin)
    urine.coalesce(1).write.option("header", "true").option("sep", ",").mode("overwrite").csv("data/feature_csv/urine")

    // serum urea nitrogen Level
    val ni = sqlContext.sql(
      """
        SELECT subject_id, hadm_id, charttime, value from LABEVENTS WHERE itemid = 51006
        """.stripMargin)
    ni.coalesce(1).write.option("header", "true").option("sep", ",").mode("overwrite").csv("data/feature_csv/nitrogen")

    // white blood cells count
    val white_blood = sqlContext.sql(
      """
        SELECT subject_id, hadm_id, charttime, value from LABEVENTS WHERE itemid = 51300 OR itemid = 51301
        """.stripMargin)
    white_blood.coalesce(1).write.option("header", "true").option("sep", ",").mode("overwrite").csv("data/feature_csv/white_blood")

    // serum bicarbonate Level
    val bicarbonate = sqlContext.sql(
      """
        SELECT subject_id, hadm_id, charttime, value from LABEVENTS WHERE itemid = 50882
        """.stripMargin)
    bicarbonate.coalesce(1).write.option("header", "true").option("sep", ",").mode("overwrite").csv("data/feature_csv/bicarbonate")

    // sodium Level
    val sodium = sqlContext.sql(
      """
        SELECT subject_id, hadm_id, charttime, value from LABEVENTS WHERE itemid = 950824 OR itemid = 50983
        """.stripMargin)
    sodium.coalesce(1).write.option("header", "true").option("sep", ",").mode("overwrite").csv("data/feature_csv/sodium")

    // potassium Level
    val potassium = sqlContext.sql(
      """
        SELECT subject_id, hadm_id, charttime, value from LABEVENTS WHERE itemid = 50822 OR itemid = 50971
        """.stripMargin)
    potassium.coalesce(1).write.option("header", "true").option("sep", ",").mode("overwrite").csv("data/feature_csv/potassium")

    // bilirubin Level
    val bilirubin = sqlContext.sql(
      """
        SELECT subject_id, hadm_id, charttime, value from LABEVENTS WHERE itemid = 50885
        """.stripMargin)
    bilirubin.coalesce(1).write.option("header", "true").option("sep", ",").mode("overwrite").csv("data/feature_csv/bilirubin")

    // age is included at icustays

    // acquired immunodeficiency syndrome (AIDS)
    val aids = sqlContext.sql(
      """
        |select distinct(subject_id) from
        |(
        |select subject_id, hadm_id, seq_num
        |, cast(icd9_code as char(5)) as icd9_code
        |from DIAGNOSES_ICD
        |) icd where icd9_code between '042  ' and '0449 '
        """.stripMargin)
    aids.coalesce(1).write.option("header", "true").option("sep", ",").mode("overwrite").csv("data/feature_csv/aids")

    // hematologic malignancy
    val hem = sqlContext.sql(
      """
        |select distinct(subject_id) from
        |(
        |select subject_id, hadm_id, seq_num
        |, cast(icd9_code as char(5)) as icd9_code
        |from DIAGNOSES_ICD
        |) icd where icd9_code between '20000' and '20238' OR
        | icd9_code between '20240' and '20248' OR -- leukemia
        | icd9_code between '20250' and '20302' OR -- lymphoma
        | icd9_code between '20310' and '20312' OR -- leukemia
        | icd9_code between '20302' and '20382' OR -- lymphoma
        | icd9_code between '20400' and '20522' OR -- chronic leukemia
        | icd9_code between '20580' and '20702' OR -- other myeloid leukemia
        | icd9_code between '20720' and '20892' OR -- other myeloid leukemia
        | icd9_code = '2386 ' OR -- lymphoma
        | icd9_code = '2733 ' -- lymphoma
        """.stripMargin)
    hem.coalesce(1).write.option("header", "true").option("sep", ",").mode("overwrite").csv("data/feature_csv/hem")

    // metastatic cancer
    val mets = sqlContext.sql(
      """
        |select distinct(subject_id) from
        |(
        |select subject_id, hadm_id, seq_num
        |, cast(icd9_code as char(5)) as icd9_code
        |from DIAGNOSES_ICD
        |) icd where icd9_code between '1960 ' and '1991 ' OR
        | icd9_code between '20970' and '20975' OR
        | icd9_code = '20979' OR
        | icd9_code = '78951'
        """.stripMargin)
    mets.coalesce(1).write.option("header", "true").option("sep", ",").mode("overwrite").csv("data/feature_csv/mets")

    // admission type
    val admission_type = sqlContext.sql(
      """
        |with surgflag as
        |(
        |select adm.hadm_id
          |, case when lower(curr_service) like '%surg%' then 1 else 0 end as surgical
          |, ROW_NUMBER() over
          |(
          |PARTITION BY adm.HADM_ID
          |ORDER BY TRANSFERTIME
          |) as serviceOrder
        |from ADMISSIONS adm
        |left join SERVICES se
        |on adm.hadm_id = se.hadm_id
        |), cohort as
        |(
        |select ie.subject_id,
    	    |case
                |when adm.ADMISSION_TYPE = 'ELECTIVE' and sf.surgical = 1
                |then 'ScheduledSurgical'
                |when adm.ADMISSION_TYPE != 'ELECTIVE' and sf.surgical = 1
                |then 'UnscheduledSurgical'
                |else 'Medical'
             |end as AdmissionType
        |from ICUSTAYS ie
        |inner join ADMISSIONS adm
        |on ie.hadm_id = adm.hadm_id
        |left join surgflag sf
        |on adm.hadm_id = sf.hadm_id and sf.serviceOrder = 1
        |)
        |select subject_id, AdmissionType from cohort
        """.stripMargin)
    admission_type.coalesce(1).write.option("header", "true").option("sep", ",").mode("overwrite").csv("data/feature_csv/admission_type")

    patients
  }

}
