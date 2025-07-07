#pragma once
#include <Control/DefaultEffectItem.hpp>
#include <Effect/EffectFactory.hpp>
#include <Process/GenericProcessFactory.hpp>

#include <QVKRT/Process.hpp>

namespace QVKRT
{
using LayerFactory = Process::GenericDefaultLayerFactory<QVKRT::Model>;
}
