#include <immersx/coral/register.h>

CORAL_PLUGIN_EXPORT int
coral_load_plugin(const char *subjson, const CoralLogger *logger)
{
  return ImmersX::Coral::load_plugin<2>(subjson, logger);
}

CORAL_PLUGIN_EXPORT void
coral_unload_plugin()
{
  ImmersX::Coral::unload_plugin<2>();
}

CORAL_PLUGIN_EXPORT const char *
coral_plugin_name()
{
  static const std::string name = ImmersX::Coral::plugin_name<2>();
  return name.c_str();
}
