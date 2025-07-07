#include "score_addon_qvkrt.hpp"

#include <score/plugins/FactorySetup.hpp>

#include <QVKRT/Executor.hpp>
#include <QVKRT/Layer.hpp>
#include <QVKRT/Process.hpp>
#include <score_plugin_engine.hpp>
#include <score_plugin_gfx.hpp>

score_addon_qvkrt::score_addon_qvkrt() { }

score_addon_qvkrt::~score_addon_qvkrt() { }

std::vector<score::InterfaceBase*>
score_addon_qvkrt::factories(
    const score::ApplicationContext& ctx,
    const score::InterfaceKey& key) const
{
  return instantiate_factories<
      score::ApplicationContext,
      FW<Process::ProcessModelFactory, QVKRT::ProcessFactory>,
      FW<Process::LayerFactory, QVKRT::LayerFactory>,
      FW<Execution::ProcessComponentFactory,
         QVKRT::ProcessExecutorComponentFactory>>(ctx, key);
}

auto score_addon_qvkrt::required() const -> std::vector<score::PluginKey>
{
  return {score_plugin_gfx::static_key()};
}

#include <score/plugins/PluginInstances.hpp>
SCORE_EXPORT_PLUGIN(score_addon_qvkrt)
