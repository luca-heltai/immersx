PR: #492

- `apps/metric_flow_x_kinsol.h` compiles with SUNDIALS 7: it uses `dealii::SUNDIALS::realtype` and `SUN_SUCCESS`, and skips `KINSetPrintLevel`, which SUNDIALS 7 removed. The KINSOL print flag has no effect there.
