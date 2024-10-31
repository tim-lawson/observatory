'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { toast } from '@/hooks/use-toast';
import { BASE_URL } from '@/app/constants';

interface LayerNeuronStats {
  total_neurons: number;
  neurons_with_descriptions: number;
  neurons_with_exemplars: number;
  neurons_with_both: number;
}

interface NeuronStats {
  total: LayerNeuronStats;
  layers: { [key: number]: LayerNeuronStats };
}

export default function NeuronStatsPage() {
  const [stats, setStats] = useState<NeuronStats | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await axios.get<NeuronStats>(
          `${BASE_URL}/neurons/stats`
        );
        setStats(response.data);
      } catch (err) {
        console.error(err);
        toast({
          title: 'Error fetching neuron stats',
          description: 'Please try again later',
        });
      }
    };

    fetchStats();
  }, []);

  const renderStatsCard = (title: string, value: number, total: number) => (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-xl font-bold">{value}</p>
        <p className="text-xs text-muted-foreground">
          {((value / total) * 100).toFixed(2)}% of total
        </p>
      </CardContent>
    </Card>
  );

  return (
    <div className="p-2">
      <h1 className="text-xl font-bold mb-2">Neuron Database Statistics</h1>

      <h2 className="text-lg font-semibold mt-4 mb-2">Overall Statistics</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {renderStatsCard(
          'Total Neurons',
          stats?.total.total_neurons || 0,
          stats?.total.total_neurons || 0
        )}
        {renderStatsCard(
          'With Descriptions',
          stats?.total.neurons_with_descriptions || 0,
          stats?.total.total_neurons || 0
        )}
        {renderStatsCard(
          'With Exemplars',
          stats?.total.neurons_with_exemplars || 0,
          stats?.total.total_neurons || 0
        )}
        {renderStatsCard(
          'With Both',
          stats?.total.neurons_with_both || 0,
          stats?.total.total_neurons || 0
        )}
      </div>

      <h2 className="text-lg font-semibold mt-4 mb-2">Layer-wise Statistics</h2>
      {Object.entries(stats?.layers || {}).map(([layer, layerStats]) => (
        <div key={layer} className="mb-4">
          <h3 className="text-base font-semibold mb-1">Layer {layer}</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {renderStatsCard(
              'Total Neurons',
              layerStats.total_neurons,
              layerStats.total_neurons
            )}
            {renderStatsCard(
              'With Descriptions',
              layerStats.neurons_with_descriptions,
              layerStats.total_neurons
            )}
            {renderStatsCard(
              'With Exemplars',
              layerStats.neurons_with_exemplars,
              layerStats.total_neurons
            )}
            {renderStatsCard(
              'With Both',
              layerStats.neurons_with_both,
              layerStats.total_neurons
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
