using System;
using System.Runtime.InteropServices;

[ComImport, Guid("56a86897-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IReferenceClock {
    [PreserveSig] int GetTime(out long timestamp);
}

class HostClock {
    static void Main() {
        var type = Type.GetTypeFromCLSID(new Guid("e436ebb1-524f-11ce-9f53-0020af0ba770"));
        var clock = (IReferenceClock)Activator.CreateInstance(type);
        while (Console.ReadLine() != null) {
            long timestamp;
            int hr = clock.GetTime(out timestamp);
            if (hr < 0) Marshal.ThrowExceptionForHR(hr);
            Console.WriteLine(timestamp);
            Console.Out.Flush();
        }
    }
}
