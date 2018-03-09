# homework1-kaladharusc ( USC ID: 7761016469 )
## 1) The LASSO and Boosting for Regression
### a) Download and read data
-   Downloaded and read data using pandas
- > Shape 1994 X 128 

    |   | state | county | community | communityname | fold | population | householdsize | racepctblack | racePctWhite | racePctAsian | racePctHisp | agePct12t21 | agePct12t29 | agePct16t24 | agePct65up | numbUrban | pctUrban | medIncome | pctWWage | pctWFarmSelf | pctWInvInc | pctWSocSec | pctWPubAsst | pctWRetire | medFamInc | perCapInc | whitePerCap | blackPerCap | indianPerCap | AsianPerCap | OtherPerCap | HispPerCap | NumUnderPov | PctPopUnderPov | PctLess9thGrade | PctNotHSGrad | PctBSorMore | PctUnemployed | PctEmploy | PctEmplManu | PctEmplProfServ | PctOccupManu | PctOccupMgmtProf | MalePctDivorce | MalePctNevMarr | FemalePctDiv | TotalPctDiv | PersPerFam | PctFam2Par | PctKids2Par | PctYoungKids2Par | PctTeen2Par | PctWorkMomYoungKids | PctWorkMom | NumIlleg | PctIlleg | NumImmig | PctImmigRecent | PctImmigRec5 | PctImmigRec8 | PctImmigRec10 | PctRecentImmig | PctRecImmig5 | PctRecImmig8 | PctRecImmig10 | PctSpeakEnglOnly | PctNotSpeakEnglWell | PctLargHouseFam | PctLargHouseOccup | PersPerOccupHous | PersPerOwnOccHous | PersPerRentOccHous | PctPersOwnOccup | PctPersDenseHous | PctHousLess3BR | MedNumBR | HousVacant | PctHousOccup | PctHousOwnOcc | PctVacantBoarded | PctVacMore6Mos | MedYrHousBuilt | PctHousNoPhone | PctWOFullPlumb | OwnOccLowQuart | OwnOccMedVal | OwnOccHiQuart | RentLowQ | RentMedian | RentHighQ | MedRent | MedRentPctHousInc | MedOwnCostPctInc | MedOwnCostPctIncNoMtg | NumInShelters | NumStreet | PctForeignBorn | PctBornSameState | PctSameHouse85 | PctSameCity85 | PctSameState85 | LemasSwornFT | LemasSwFTPerPop | LemasSwFTFieldOps | LemasSwFTFieldPerPop | LemasTotalReq | LemasTotReqPerPop | PolicReqPerOffic | PolicPerPop | RacialMatchCommPol | PctPolicWhite | PctPolicBlack | PctPolicHisp | PctPolicAsian | PctPolicMinor | OfficAssgnDrugUnits | NumKindsDrugsSeiz | PolicAveOTWorked | LandArea | PopDens | PctUsePubTrans | PolicCars | PolicOperBudg | LemasPctPolicOnPatr | LemasGangUnitDeploy | LemasPctOfficDrugUn | PolicBudgPerPop | ViolentCrimesPerPop |
    |---|-------|--------|-----------|---------------|------|------------|---------------|--------------|--------------|--------------|-------------|-------------|-------------|-------------|------------|-----------|----------|-----------|----------|--------------|------------|------------|-------------|------------|-----------|-----------|-------------|-------------|--------------|-------------|-------------|------------|-------------|----------------|-----------------|--------------|-------------|---------------|-----------|-------------|-----------------|--------------|------------------|----------------|----------------|--------------|-------------|------------|------------|-------------|------------------|-------------|---------------------|------------|----------|----------|----------|----------------|--------------|--------------|---------------|----------------|--------------|--------------|---------------|------------------|---------------------|-----------------|-------------------|------------------|-------------------|--------------------|-----------------|------------------|----------------|----------|------------|--------------|---------------|------------------|----------------|----------------|----------------|----------------|----------------|--------------|---------------|----------|------------|-----------|---------|-------------------|------------------|-----------------------|---------------|-----------|----------------|------------------|----------------|---------------|----------------|--------------|-----------------|-------------------|----------------------|---------------|-------------------|------------------|-------------|--------------------|---------------|---------------|--------------|---------------|---------------|---------------------|-------------------|------------------|----------|---------|----------------|-----------|---------------|---------------------|---------------------|---------------------|-----------------|---------------------|
    | 0 | 8     | ?      | ?         | Lakewoodcity  | 1    | 0.19       | 0.33          | 0.02         | 0.9          | 0.12         | 0.17        | 0.34        | 0.47        | 0.29        | 0.32       | 0.2       | 1.0      | 0.37      | 0.72     | 0.34         | 0.6        | 0.29       | 0.15        | 0.43       | 0.39      | 0.4       | 0.39        | 0.32        | 0.27         | 0.27        | 0.36        | 0.41       | 0.08        | 0.19           | 0.1             | 0.18         | 0.48        | 0.27          | 0.68      | 0.23        | 0.41            | 0.25         | 0.52             | 0.68           | 0.4            | 0.75         | 0.75        | 0.35       | 0.55       | 0.59        | 0.61             | 0.56        | 0.74                | 0.76       | 0.04     | 0.14     | 0.03     | 0.24           | 0.27         | 0.37         | 0.39          | 0.07           | 0.07         | 0.08         | 0.08          | 0.89             | 0.06                | 0.14            | 0.13              | 0.33             | 0.39              | 0.28               | 0.55            | 0.09             | 0.51           | 0.5      | 0.21       | 0.71         | 0.52          | 0.05             | 0.26           | 0.65           | 0.14           | 0.06           | 0.22           | 0.19         | 0.18          | 0.36     | 0.35       | 0.38      | 0.34    | 0.38              | 0.46             | 0.25                  | 0.04          | 0.0       | 0.12           | 0.42             | 0.5            | 0.51          | 0.64           | 0.03         | 0.13            | 0.96              | 0.17                 | 0.06          | 0.18              | 0.44             | 0.13        | 0.94               | 0.93          | 0.03          | 0.07         | 0.1           | 0.07          | 0.02                | 0.57              | 0.29             | 0.12     | 0.26    | 0.2            | 0.06      | 0.04          | 0.9                 | 0.5                 | 0.32                | 0.14            | 0.2                 |
    | 1 | 53    | ?      | ?         | Tukwilacity   | 1    | 0.0        | 0.16          | 0.12         | 0.74         | 0.45         | 0.07        | 0.26        | 0.59        | 0.35        | 0.27       | 0.02      | 1.0      | 0.31      | 0.72     | 0.11         | 0.45       | 0.25       | 0.29        | 0.39       | 0.29      | 0.37      | 0.38        | 0.33        | 0.16         | 0.3         | 0.22        | 0.35       | 0.01        | 0.24           | 0.14            | 0.24         | 0.3         | 0.27          | 0.73      | 0.57        | 0.15            | 0.42         | 0.36             | 1.0            | 0.63           | 0.91         | 1.0         | 0.29       | 0.43       | 0.47        | 0.6              | 0.39        | 0.46                | 0.53       | 0.0      | 0.24     | 0.01     | 0.52           | 0.62         | 0.64         | 0.63          | 0.25           | 0.27         | 0.25         | 0.23          | 0.84             | 0.1                 | 0.16            | 0.1               | 0.17             | 0.29              | 0.17               | 0.26            | 0.2              | 0.82           | 0.0      | 0.02       | 0.79         | 0.24          | 0.02             | 0.25           | 0.65           | 0.16           | 0.0            | 0.21           | 0.2          | 0.21          | 0.42     | 0.38       | 0.4       | 0.37    | 0.29              | 0.32             | 0.18                  | 0.0           | 0.0       | 0.21           | 0.5              | 0.34           | 0.6           | 0.52           | ?            | ?               | ?                 | ?                    | ?             | ?                 | ?                | ?           | ?                  | ?             | ?             | ?            | ?             | ?             | ?                   | ?                 | ?                | 0.02     | 0.12    | 0.45           | ?         | ?             | ?                   | ?                   | 0.0                 | ?               | 0.67                |
    
- > As we can see data has some missing values.

### b) Use Data Imputation Techniques.
-   Filled missing Values with their coresponding column means
- > Shape 1994 X 124 ( removed unpredictors )
     
    |   | fold | population | householdsize | racepctblack | racePctWhite | racePctAsian | racePctHisp | agePct12t21 | agePct12t29 | agePct16t24 | agePct65up | numbUrban | pctUrban | medIncome | pctWWage | pctWFarmSelf | pctWInvInc | pctWSocSec | pctWPubAsst | pctWRetire | medFamInc | perCapInc | whitePerCap | blackPerCap | indianPerCap | AsianPerCap | OtherPerCap | HispPerCap | NumUnderPov | PctPopUnderPov | PctLess9thGrade | PctNotHSGrad | PctBSorMore | PctUnemployed | PctEmploy | PctEmplManu | PctEmplProfServ | PctOccupManu | PctOccupMgmtProf | MalePctDivorce | MalePctNevMarr | FemalePctDiv | TotalPctDiv | PersPerFam | PctFam2Par | PctKids2Par | PctYoungKids2Par | PctTeen2Par | PctWorkMomYoungKids | PctWorkMom | NumIlleg | PctIlleg | NumImmig | PctImmigRecent | PctImmigRec5 | PctImmigRec8 | PctImmigRec10 | PctRecentImmig | PctRecImmig5 | PctRecImmig8 | PctRecImmig10 | PctSpeakEnglOnly | PctNotSpeakEnglWell | PctLargHouseFam | PctLargHouseOccup | PersPerOccupHous | PersPerOwnOccHous | PersPerRentOccHous | PctPersOwnOccup | PctPersDenseHous | PctHousLess3BR | MedNumBR | HousVacant | PctHousOccup | PctHousOwnOcc | PctVacantBoarded | PctVacMore6Mos | MedYrHousBuilt | PctHousNoPhone | PctWOFullPlumb | OwnOccLowQuart | OwnOccMedVal | OwnOccHiQuart | RentLowQ | RentMedian | RentHighQ | MedRent | MedRentPctHousInc | MedOwnCostPctInc | MedOwnCostPctIncNoMtg | NumInShelters | NumStreet | PctForeignBorn | PctBornSameState | PctSameHouse85 | PctSameCity85 | PctSameState85 | LemasSwornFT        | LemasSwFTPerPop     | LemasSwFTFieldOps  | LemasSwFTFieldPerPop | LemasTotalReq       | LemasTotReqPerPop   | PolicReqPerOffic    | PolicPerPop         | RacialMatchCommPol | PctPolicWhite      | PctPolicBlack       | PctPolicHisp        | PctPolicAsian     | PctPolicMinor       | OfficAssgnDrugUnits | NumKindsDrugsSeiz | PolicAveOTWorked  | LandArea | PopDens | PctUsePubTrans | PolicCars           | PolicOperBudg       | LemasPctPolicOnPatr | LemasGangUnitDeploy | LemasPctOfficDrugUn | PolicBudgPerPop    | ViolentCrimesPerPop |
    |---|------|------------|---------------|--------------|--------------|--------------|-------------|-------------|-------------|-------------|------------|-----------|----------|-----------|----------|--------------|------------|------------|-------------|------------|-----------|-----------|-------------|-------------|--------------|-------------|-------------|------------|-------------|----------------|-----------------|--------------|-------------|---------------|-----------|-------------|-----------------|--------------|------------------|----------------|----------------|--------------|-------------|------------|------------|-------------|------------------|-------------|---------------------|------------|----------|----------|----------|----------------|--------------|--------------|---------------|----------------|--------------|--------------|---------------|------------------|---------------------|-----------------|-------------------|------------------|-------------------|--------------------|-----------------|------------------|----------------|----------|------------|--------------|---------------|------------------|----------------|----------------|----------------|----------------|----------------|--------------|---------------|----------|------------|-----------|---------|-------------------|------------------|-----------------------|---------------|-----------|----------------|------------------|----------------|---------------|----------------|---------------------|---------------------|--------------------|----------------------|---------------------|---------------------|---------------------|---------------------|--------------------|--------------------|---------------------|---------------------|-------------------|---------------------|---------------------|-------------------|-------------------|----------|---------|----------------|---------------------|---------------------|---------------------|---------------------|---------------------|--------------------|---------------------|
    | 0 | 1    | 0.19       | 0.33          | 0.02         | 0.9          | 0.12         | 0.17        | 0.34        | 0.47        | 0.29        | 0.32       | 0.2       | 1.0      | 0.37      | 0.72     | 0.34         | 0.6        | 0.29       | 0.15        | 0.43       | 0.39      | 0.4       | 0.39        | 0.32        | 0.27         | 0.27        | 0.36        | 0.41       | 0.08        | 0.19           | 0.1             | 0.18         | 0.48        | 0.27          | 0.68      | 0.23        | 0.41            | 0.25         | 0.52             | 0.68           | 0.4            | 0.75         | 0.75        | 0.35       | 0.55       | 0.59        | 0.61             | 0.56        | 0.74                | 0.76       | 0.04     | 0.14     | 0.03     | 0.24           | 0.27         | 0.37         | 0.39          | 0.07           | 0.07         | 0.08         | 0.08          | 0.89             | 0.06                | 0.14            | 0.13              | 0.33             | 0.39              | 0.28               | 0.55            | 0.09             | 0.51           | 0.5      | 0.21       | 0.71         | 0.52          | 0.05             | 0.26           | 0.65           | 0.14           | 0.06           | 0.22           | 0.19         | 0.18          | 0.36     | 0.35       | 0.38      | 0.34    | 0.38              | 0.46             | 0.25                  | 0.04          | 0.0       | 0.12           | 0.42             | 0.5            | 0.51          | 0.64           | 0.03                | 0.13                | 0.96               | 0.17                 | 0.06                | 0.18                | 0.44                | 0.13                | 0.94               | 0.93               | 0.03                | 0.07                | 0.1               | 0.07                | 0.02                | 0.57              | 0.29              | 0.12     | 0.26    | 0.2            | 0.06                | 0.04                | 0.9                 | 0.5                 | 0.32                | 0.14               | 0.2                 |
    | 1 | 1    | 0.0        | 0.16          | 0.12         | 0.74         | 0.45         | 0.07        | 0.26        | 0.59        | 0.35        | 0.27       | 0.02      | 1.0      | 0.31      | 0.72     | 0.11         | 0.45       | 0.25       | 0.29        | 0.39       | 0.29      | 0.37      | 0.38        | 0.33        | 0.16         | 0.3         | 0.22        | 0.35       | 0.01        | 0.24           | 0.14            | 0.24         | 0.3         | 0.27          | 0.73      | 0.57        | 0.15            | 0.42         | 0.36             | 1.0            | 0.63           | 0.91         | 1.0         | 0.29       | 0.43       | 0.47        | 0.6              | 0.39        | 0.46                | 0.53       | 0.0      | 0.24     | 0.01     | 0.52           | 0.62         | 0.64         | 0.63          | 0.25           | 0.27         | 0.25         | 0.23          | 0.84             | 0.1                 | 0.16            | 0.1               | 0.17             | 0.29              | 0.17               | 0.26            | 0.2              | 0.82           | 0.0      | 0.02       | 0.79         | 0.24          | 0.02             | 0.25           | 0.65           | 0.16           | 0.0            | 0.21           | 0.2          | 0.21          | 0.42     | 0.38       | 0.4       | 0.37    | 0.29              | 0.32             | 0.18                  | 0.0           | 0.0       | 0.21           | 0.5              | 0.34           | 0.6           | 0.52           | 0.06965517241379311 | 0.21746081504702197 | 0.9247335423197492 | 0.2463322884012539   | 0.09799373040752352 | 0.21520376175548586 | 0.34363636363636363 | 0.21749216300940438 | 0.6894043887147336 | 0.7269592476489029 | 0.22047021943573672 | 0.13485893416927902 | 0.114858934169279 | 0.25918495297805644 | 0.0755485893416928  | 0.556050156739812 | 0.305987460815047 | 0.02     | 0.12    | 0.45           | 0.16310344827586207 | 0.07670846394984326 | 0.69858934169279    | 0.44043887147335425 | 0.0                 | 0.1950783699059561 | 0.67                |
- > We can see there are no missing values

### c) Correlation Matrix
-   ![alt text](https://github.com/MLforDTIN-18Srping/homework3-kaladharusc/blob/master/plots/1_c_correlations.png "correlation matrix")

### d) Coefficient of Variation for each feature.
-   Calculated using `Standard Deviation / Mean` .

    | Feature               | Coefficient Of Variation |
    |-----------------------|--------------------------|
    | NumStreet             | 4.292922989491593        |
    | NumInShelters         | 3.470952139705214        |
    | NumIlleg              | 3.0589643472092356       |
    | NumImmig              | 2.9266352462888148       |
    | LemasPctOfficDrugUn   | 2.552945511727576        |
    | NumUnderPov           | 2.3424431162181505       |
    | population            | 2.2411046245803745       |
    | numbUrban             | 2.0384614919156445       |
    | HousVacant            | 1.9684670491351257       |
    | LandArea              | 1.6454078602149063       |
    | racePctHisp           | 1.612091005228411        |
    | PctNotSpeakEnglWell   | 1.4566183675039828       |
    | racepctblack          | 1.4288854186502822       |
    | PctUsePubTrans        | 1.3971097612127603       |
    | racePctAsian          | 1.359099684159002        |
    | PctRecentImmig        | 1.288286991787815        |
    | PctRecImmig5          | 1.286184069121448        |
    | PctRecImmig10         | 1.2704761039803925       |
    | PctRecImmig8          | 1.2655788622333226       |
    | PctPersDenseHous      | 1.1329256518367359       |
    | PctForeignBorn        | 1.0658797341302004       |
    | PctVacantBoarded      | 1.0548273380808388       |
    | ViolentCrimesPerPop   | 0.9879088645521366       |
    | PctHousNoPhone        | 0.9227647590220344       |
    | PctIlleg              | 0.9215499866941839       |
    | OwnOccMedVal          | 0.889409997564829        |
    | OwnOccHiQuart         | 0.8835825993723008       |
    | LemasSwornFT          | 0.8722529356778985       |
    | PopDens               | 0.8644992034803372       |
    | OwnOccLowQuart        | 0.8595655789242966       |
    | PctWOFullPlumb        | 0.8432062676593917       |
    | indianPerCap          | 0.8183681264327733       |
    | MedNumBR              | 0.8139470895341551       |
    | PolicOperBudg         | 0.7743443211261958       |
    | PctLargHouseOccup     | 0.7631976222314811       |
    | PctPopUnderPov        | 0.754469978085357        |
    | PctLargHouseFam       | 0.7386915454231807       |
    | PctPolicAsian         | 0.7262293163276773       |
    | pctWPubAsst           | 0.7062058943769388       |
    | pctWFarmSelf          | 0.701897873082006        |
    | LemasTotalReq         | 0.6975275179267774       |
    | OfficAssgnDrugUnits   | 0.6888091316674114       |
    | PctImmigRecent        | 0.6814864307426184       |
    | PctLess9thGrade       | 0.6804903296377428       |
    | OtherPerCap           | 0.6732169975302158       |
    | pctUrban              | 0.6433181716461008       |
    | RentLowQ              | 0.6396616452190297       |
    | AsianPerCap           | 0.6148655265869314       |
    | blackPerCap           | 0.5993405451897994       |
    | PctPolicHisp          | 0.5949533571476963       |
    | RentHighQ             | 0.5940594809708987       |
    | PctBSorMore           | 0.5926798349812531       |
    | medIncome             | 0.5919781653450651       |
    | PctImmigRec5          | 0.5907934182085816       |
    | RentMedian            | 0.5687776924869173       |
    | MedRent               | 0.5634425038390524       |
    | PctUnemployed         | 0.562424698851533        |
    | perCapInc             | 0.5580794526579645       |
    | medFamInc             | 0.5406701530549352       |
    | PolicCars             | 0.5398226917021794       |
    | PctNotHSGrad          | 0.5337533808931807       |
    | whitePerCap           | 0.519828467024789        |
    | PctOccupManu          | 0.5163793587769291       |
    | PctEmplManu           | 0.5135297542547989       |
    | fold                  | 0.510302111015002        |
    | PctImmigRec8          | 0.5050721810572517       |
    | agePct16t24           | 0.5003099115377213       |
    | HispPerCap            | 0.4840679504501606       |
    | PersPerRentOccHous    | 0.4736550685710704       |
    | MedYrHousBuilt        | 0.470762074441811        |
    | MedOwnCostPctIncNoMtg | 0.4702069718356742       |
    | PctImmigRec10         | 0.4577484543862395       |
    | PctPolicBlack         | 0.4472186954207617       |
    | PctVacMore6Mos        | 0.43808176451988246      |
    | PctOccupMgmtProf      | 0.43266953598425273      |
    | MedOwnCostPctInc      | 0.41550160572977074      |
    | agePct65up            | 0.4127759758512281       |
    | MalePctNevMarr        | 0.40406152224265024      |
    | PctEmplProfServ       | 0.40284884867313686      |
    | MalePctDivorce        | 0.3993660859242948       |
    | TotalPctDiv           | 0.37390450168497985      |
    | LemasGangUnitDeploy   | 0.3710015895528759       |
    | agePct12t21           | 0.3690831192095117       |
    | PersPerOccupHous      | 0.3672534015041071       |
    | pctWInvInc            | 0.365334582744191        |
    | pctWSocSec            | 0.36071544651696696      |
    | FemalePctDiv          | 0.3603898277589537       |
    | PctPolicMinor         | 0.356241433781327        |
    | householdsize         | 0.3557995664028941       |
    | PctEmploy             | 0.3514504046780556       |
    | PctPersOwnOccup       | 0.34920959278252445      |
    | pctWRetire            | 0.3454209567630026       |
    | PctHousLess3BR        | 0.34355333525689935      |
    | MedRentPctHousInc     | 0.34306486918919427      |
    | PctWorkMomYoungKids   | 0.3406809390881452       |
    | PctBornSameState      | 0.33624489274009706      |
    | PctHousOwnOcc         | 0.33612858762074044      |
    | PctWorkMom            | 0.3360178908039217       |
    | PctKids2Par           | 0.33487556916187433      |
    | PctSameHouse85        | 0.33451877905492433      |
    | PctYoungKids2Par      | 0.3338334680337065       |
    | PctFam2Par            | 0.3331946554395547       |
    | racePctWhite          | 0.3302126634442826       |
    | pctWWage              | 0.3286140569015946       |
    | PctTeen2Par           | 0.3278008334193768       |
    | PolicBudgPerPop       | 0.32003502145419926      |
    | PersPerOwnOccHous     | 0.3192144507431373       |
    | PctSameCity85         | 0.31875804915193473      |
    | PersPerFam            | 0.31737540470857095      |
    | LemasTotReqPerPop     | 0.3094225266056299       |
    | PolicAveOTWorked      | 0.30888779112006637      |
    | PctSameState85        | 0.2984340732673321       |
    | agePct12t29           | 0.2913148106352062       |
    | PctSpeakEnglOnly      | 0.28999867240218546      |
    | LemasSwFTPerPop       | 0.28843327513706196      |
    | PolicPerPop           | 0.2883990936086546       |
    | PctHousOccup          | 0.2681815038662207       |
    | LemasSwFTFieldPerPop  | 0.24500189758872482      |
    | PolicReqPerOffic      | 0.2263593215884439       |
    | NumKindsDrugsSeiz     | 0.14353283569192646      |
    | RacialMatchCommPol    | 0.13373712221130316      |
    | PctPolicWhite         | 0.12258236559333853      |
    | LemasPctPolicOnPatr   | 0.11798844847958818      |
    | LemasSwFTFieldOps     | 0.06400225509556129      |
    
### e) Scatter plot and Box plots
-   ![alt text](https://github.com/MLforDTIN-18Srping/homework3-kaladharusc/blob/master/plots/1_e_scatter_plots.png "pairwise scatter plot")
-   ![alt text](https://github.com/MLforDTIN-18Srping/homework3-kaladharusc/blob/master/plots/1_e_scatter_plots_response.png "response feature scatter plot")
-   ![alt text](https://github.com/MLforDTIN-18Srping/homework3-kaladharusc/blob/master/plots/1_e_boxplot.png "box plot")
-   ~TODO: Conclistions~

### f) Linear Regression:

   > **Test Mean Squared Error: 0.7716253161613343**

### g) Ridge Regression:

   > **Lambda : 1.0**  
   > **Test Mean Squared Error : 0.0177**