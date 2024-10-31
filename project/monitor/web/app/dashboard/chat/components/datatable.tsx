'use client';

import * as React from 'react';
import {
  ColumnDef,
  SortingState,
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import { useSelector } from 'react-redux';
import store, { RootState } from '@/app/store/store';

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Neuron, NeuronPolarity } from '@/app/types/neuronData';
import { Button } from '@/components/ui/button'; // Import Button component
import { cn } from '@/lib/utils';
import {
  Tooltip,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { TooltipContent } from '@radix-ui/react-tooltip';
import { usePostHog } from 'posthog-js/react';

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
  isLoading?: boolean; // Add isLoading prop
  sortingState?: SortingState;
  onMouseOverRow?: (neuron?: Neuron) => void;
}

export const DEFAULT_ACTIVATION_SORTING_STATE = [
  { id: 'isInteresting', desc: true },
  { id: 'activationNormalized', desc: true },
];
export const DEFAULT_ATTRIBUTION_SORTING_STATE = [
  { id: 'isInteresting', desc: true },
  { id: 'attribution', desc: true },
];

export function DataTable<TData, TValue>({
  columns,
  data,
  isLoading = false, // Default to false if not provided
  sortingState,
  onMouseOverRow, // Add onMouseOverRow prop
}: DataTableProps<TData, TValue>) {
  // Allow external components to change sorting state
  const [sorting, setSorting] = React.useState<SortingState>(
    sortingState || DEFAULT_ACTIVATION_SORTING_STATE
  );
  React.useEffect(() => {
    setSorting(sortingState || DEFAULT_ACTIVATION_SORTING_STATE);
  }, [sortingState]);

  const [pagination, setPagination] = React.useState({
    pageIndex: 0,
    pageSize: 50,
  });
  const [shouldScrollToCluster, setShouldScrollToCluster] =
    React.useState(false);

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    onSortingChange: setSorting,
    getSortedRowModel: getSortedRowModel(),
    onPaginationChange: setPagination,
    state: {
      sorting,
      pagination,
    },
    enableMultiSort: true,
  });

  const posthog = usePostHog();
  const sessionId = useSelector((state: RootState) => state.uiState.sessionId);

  const handleRowClick = (
    layer: number,
    neuron: number,
    polarity: NeuronPolarity
  ) => {
    // Track row click event
    posthog.capture('neuron_row_clicked', {
      layer: layer,
      neuron: neuron,
      polarity: polarity,
      sessionId: sessionId,
    });
    window.open(
      `https://neurons.transluce.org/${layer}/${neuron}/${polarity === NeuronPolarity.POS ? '+' : '-'}`,
      '_blank'
    );
  };

  const firstClusterRowRef = React.useRef<HTMLTableRowElement | null>(null);

  React.useEffect(() => {
    if (data) {
      // Find the index of the first neuron in the selected cluster
      const index = data.findIndex((row: any) => row.inSelectedCluster);

      if (index >= 0) {
        const pageSize = table.getState().pagination.pageSize;
        const pageIndex = Math.floor(index / pageSize);

        // FIXME this is completely broken...
        if (table.getState().pagination.pageIndex !== pageIndex) {
          table.setPageIndex(pageIndex);
          setShouldScrollToCluster(true);
        } else {
          scrollToClusterRow();
        }
      }
    }
  }, [data, table]);

  React.useEffect(() => {
    if (shouldScrollToCluster) {
      scrollToClusterRow();
      setShouldScrollToCluster(false);
    }
  }, [table.getState().pagination.pageIndex]);

  const scrollToClusterRow = () => {
    setTimeout(() => {
      const clusterRow = document.querySelector('tr.bg-light-green');
      if (clusterRow) {
        clusterRow.scrollIntoView({
          behavior: 'smooth',
          block: 'center',
        });
      }
    }, 100);
  };

  return (
    <>
      <div className="rounded-md border flex flex-col flex-1 overflow-y-auto">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  return (
                    <TableHead key={header.id}>
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
                    </TableHead>
                  );
                })}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  data-state={row.getIsSelected() && 'selected'}
                  onClick={() =>
                    handleRowClick(
                      // @ts-ignore
                      row.original.layer,
                      // @ts-ignore
                      row.original.neuron,
                      // @ts-ignore
                      row.original.polarity
                    )
                  }
                  onMouseOver={() =>
                    onMouseOverRow &&
                    onMouseOverRow({
                      // @ts-ignore
                      layer: row.original.layer,
                      // @ts-ignore
                      neuron: row.original.neuron,
                      token: null,
                      polarity: null,
                    })
                  }
                  onMouseOut={() => onMouseOverRow && onMouseOverRow(undefined)}
                  className={cn(
                    'cursor-pointer transition-colors',
                    // @ts-ignore
                    row.original.inSelectedCluster
                      ? 'bg-light-green hover:bg-mid-green'
                      : 'hover:bg-muted/100'
                  )}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-24 text-center"
                >
                  {isLoading ? 'Loading results...' : 'No results'}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
      <div className="flex items-center justify-end space-x-2 mt-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            table.previousPage();
            // Track previous page click
            posthog.capture('datatable_previous_page', {
              currentPage: table.getState().pagination.pageIndex,
              sessionId: sessionId,
            });
          }}
          disabled={!table.getCanPreviousPage()}
        >
          Previous
        </Button>
        <span className="flex items-center gap-1 text-xs">
          <div>Page</div>
          <strong>
            {pagination.pageIndex + 1} of {Math.max(table.getPageCount(), 1)}
          </strong>
        </span>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            table.nextPage();
            // Track next page click
            posthog.capture('datatable_next_page', {
              currentPage: table.getState().pagination.pageIndex,
              sessionId: sessionId,
            });
          }}
          disabled={!table.getCanNextPage()}
        >
          Next
        </Button>
      </div>
    </>
  );
}
